import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import json
import os
from dataloader.nyu_transform import *
from random import randrange
import glob


try:
    import accimage
except ImportError:
    accimage = None


def load_annotation_data(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


def pil_loader(path):
    return Image.open(path)


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dict_dir, root_dir, img_size = [384,512], transform=None, is_test=False):
        # fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        self.height = img_size[0]
        self.width = img_size[1]
        self.root_dir = root_dir
        self.transform = transform
        self.db_seq_len = 8

        csv_files = glob.glob(os.path.join(*[dict_dir, "**/*.csv"]), recursive=True)
        self.datalines = []
        for i, file_name in enumerate(csv_files):
            with open(file_name, 'r') as file_input:
                lines = file_input.readlines()
                rest = (len(lines)-1)%self.db_seq_len
                self.datalines += lines[1+rest:]
        # print("Items count : %i" % len(self.datalines))
        if is_test:
            self.seq_len = self.db_seq_len
        else:
            self.seq_len = 5
        self.is_test = is_test

    def depth_video_loader(self, root_dir, index, rand_offset):
        video = []
        for i in range(self.seq_len):
            line = self.datalines[index+i].split("\t")
            depth_file_name = line[2]
            image_path = os.path.join(self.root_dir, depth_file_name)

            def open_depth(image_path, h, w):
                pic = Image.fromarray(np.load(image_path))
                img = pic.resize([self.width, self.height], Image.NEAREST)
                # img = transforms.Resize([h, w], interpolation=0)(pic) #nearest interpolationtransforms.functional.InterpolationMode.NEAREST)(pic)
                return img

            if os.path.exists(image_path):
                if self.is_test:
                    depth_gt = open_depth(image_path, self.height, self.width)
                else:
                    depth_gt = open_depth(image_path, self.height, self.width)
                depth_gt = np.clip(depth_gt, 0.001, 100.)
                video.append(depth_gt)
            else:
                print("File %s not found" % image_path)

        return video

    def color_video_loader(self, root_dir, index, rand_offset):
        video = []
        for i in range(self.seq_len):
            line = self.datalines[index+i].split("\t")
            file_name = line[1]
            image_path = os.path.join(self.root_dir, file_name)

            if os.path.exists(image_path):
                pic = Image.open(image_path)
                pic = pic.resize([self.width, self.height], Image.BILINEAR)
                video.append(pic)
            else:
                print("File %s not found" % image_path)

        return video

    def __getitem__(self, idx):
        if self.is_test:
            rand_offset = 0
        else:
            rand_offset = randrange(self.db_seq_len-self.seq_len)
        rgb_clips = self.color_video_loader(self.root_dir, idx*self.db_seq_len, rand_offset)
        depth_clips = self.depth_video_loader(self.root_dir, idx*self.db_seq_len, rand_offset)

        # WARNING pixels with no color information are disabled
        for i, rgm_frame in enumerate(rgb_clips):
            greyscale_frame = np.linalg.norm(rgm_frame, ord=2, axis=-1)
            depth_clips[i][greyscale_frame==0] = float('nan')

        rgb_tensor = []
        depth_tensor = []
        depth_scaled_tensor = []
        for rgb_clip, depth_clip in zip(rgb_clips, depth_clips):
            sample = {'image': rgb_clip, 'depth': depth_clip}
            sample_new = self.transform(sample)
            rgb_tensor.append(sample_new['image'])
            depth_tensor.append(sample_new['depth'])

        return torch.stack(rgb_tensor, 0).permute(1, 0, 2, 3), \
               torch.stack(depth_tensor, 0).permute(1, 0, 2, 3), \
               4

    def __len__(self):
        return len(self.datalines)//self.db_seq_len

def getTrainingData_TartanAir(batch_size=64, dict_dir=None, root_dir=None, img_size = [384,576]):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transformed_training = depthDataset(dict_dir=dict_dir,
                                        root_dir = root_dir,
                                        img_size=img_size,
                                        transform=transforms.Compose([
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'], 
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size, shuffle=True, num_workers=1, pin_memory=False)

    return dataloader_training

def getTestingData_TartanAir(batch_size=64, dict_dir=None, root_dir=None, num_workers=4, img_size = [384,576]):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}


    transformed_testing = depthDataset(dict_dir=dict_dir,
                                       root_dir=root_dir,
                                       img_size=img_size,
                                       transform=transforms.Compose([
                                            ToTensor(is_test=True),
                                            Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]),
                                       is_test=True)

    dataloader_testing = DataLoader(transformed_testing, batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return dataloader_testing
