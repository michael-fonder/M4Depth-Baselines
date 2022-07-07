import time
import torch
import numpy as np
import os
from collections import OrderedDict
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
from utils import *
from options import get_args
from dataloader import midair_dataloader, tartanair_dataloader
from models.backbone_dict import backbone_dict
from models import modules
from models import net
from PIL import Image
import os


args = get_args('test')
# lode nyud v2 test set
dir_path = os.path.dirname(os.path.realpath(__file__))
if args.dataset == "midair":
    TestImgLoader = midair_dataloader.getTestingData_MidAir(args.batch_size, os.path.join(*[dir_path, args.test_files]), args.root_path, img_size=[args.h, args.w])
elif args.dataset == "tartanair":
    TestImgLoader = tartanair_dataloader.getTestingData_TartanAir(args.batch_size, os.path.join(*[dir_path, args.test_files]), args.root_path, img_size=[args.h, args.w])
else:
    print("No matching dataset found")
# model
backbone = backbone_dict[args.backbone]()
Encoder = modules.E_resnet(backbone)

if args.backbone in ['resnet50']:
    model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], refinenet=args.refinenet)
elif args.backbone in ['resnet18', 'resnet34']:
    model = net.model(Encoder, num_features=512, block_channel=[64, 128, 256, 512], refinenet=args.refinenet)

model = nn.DataParallel(model).cuda()

# load test model
if args.loadckpt is not None and args.loadckpt.endswith('.pth.tar'):
    print("loading the specific model in checkpoint_dir: {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict)
elif os.path.isdir(args.loadckpt):
    all_saved_ckpts = [ckpt for ckpt in os.listdir(args.loadckpt) if ckpt.endswith(".pth.tar")]
    print(all_saved_ckpts)
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x:int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(args.loadckpt, all_saved_ckpts[-1])
    start_epoch = int(all_saved_ckpts[-1].split('_')[-1].split('.')[0])
    print("loading the lastest model in checkpoint_dir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict)
else:
    print("You have not loaded any models.")

def test():
    def save_maps(gt, est, id):

        def save_depth(map, path):
            map = np.clip(map, np.exp(0.), 80.)
            I8_content = (np.log(map) * 255.0/np.log(80.)).astype(np.uint8)
            img_tmp = Image.fromarray(np.stack((I8_content, I8_content, I8_content), axis=2))
            img_tmp.save(path)

        save_dir = os.path.join(args.loadckpt, "saved_maps")
        os.makedirs(save_dir, exist_ok = True)
        filename_gt = os.path.join(save_dir,  str(id).zfill(6) + "_gt.png")
        save_depth(gt, filename_gt)
        filename_est = os.path.join(save_dir,  str(id).zfill(6) + "_est.png")
        save_depth(est, filename_est)

    seq_len = args.seq_len
    model.eval()
    im_id=0
    with torch.no_grad():
        total_metrics = None
        batch_cnter = 0
        for batch_idx, sample in enumerate(TestImgLoader):
            if not batch_idx%250:
                print("Processing the {}th batch!".format(batch_idx))
            image, depth = sample[0], sample[1]
            depth = depth.cuda()
            image = image.cuda()

            image = torch.autograd.Variable(image)
            depth = torch.autograd.Variable(depth)
            
            start = time.time()
            pred = model(image)
            end = time.time()
            running_time = end - start

            metrics_avg_batch = None
            for i in range(pred.size()[0]):
                for seq_idx in range(seq_len): #5=sequence length
                    single_pred = torch.nn.Upsample(scale_factor=2, mode='nearest')(pred[i:i+1,:,seq_idx,:,:])
                    single_depth = depth[i:i+1,:,seq_idx,:,:]

                    if args.export_pics and seq_idx==seq_len-1:
                        save_maps(depth.cpu().numpy()[i,0,seq_idx,:,:], pred.cpu().numpy()[i,0,seq_idx,:,:], im_id)
                        im_id += 1

                    # single_pred = torch.clamp(single_pred, 0.001, 80.0)
                    # single_depth = torch.clamp(single_depth, 0.001, 80.0)

                    if metrics_avg_batch is None:
                        metrics_avg_batch = evaluateError(single_pred, single_depth)
                    else:
                        metrics = evaluateError(single_pred, single_depth)
                        metrics_avg_batch = addErrors(metrics_avg_batch, metrics, 1)

            batch_cnter += 1
            metrics_avg_batch = averageErrors(metrics_avg_batch, seq_len*pred.size()[0])
            if total_metrics is None:
                total_metrics = metrics_avg_batch
            else:
                total_metrics = addErrors(total_metrics, metrics_avg_batch, 1)
        avg_metrics = averageErrors(total_metrics, batch_cnter)

        metric_names = ['SQ_REL', 'RMSE', 'ABS_REL', 'RMSEl','MAE',  'DELTA1', 'DELTA2', 'DELTA3']
        print("Averaged losses")
        for name in metric_names:
            print("%s \t: %f" % (name, avg_metrics[name]))


if __name__ == '__main__':
    test()