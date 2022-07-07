import argparse

def _get_train_opt():
    parser = argparse.ArgumentParser(description = 'Monocular Depth Estimation')
    parser.add_argument('--trainlist_path', required=True, help='the path of trainlist', default='/media3/x00532679/project/ST-SARPN/data_list/raw_nyu_v2_250k/raw_nyu_v2_250k_fps30_fl5_op0_end_train.json')
    parser.add_argument('--root_path', required=True, help="the root path of dataset", default='/media3/x00532679/data/')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--dataset', choices=['midair', 'tartanair'], required=True, help='dataset to use for the test')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
    parser.add_argument('--logdir', required=True, help="the directory to save logs and checkpoints", default='./checkpoint')
    parser.add_argument('--checkpoint_dir', required=True, help="the directory to save the checkpoints", default='./log_224')
    parser.add_argument('--loadckpt', type=str) 
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--resume', action='store_true',default=False, help='continue training the model')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum parameter used in the Optimizer.')
    parser.add_argument('--epsilon', default=0.001, type=float, help='epsilon')
    parser.add_argument('--optimizer_name', default="adam", type=str, help="Optimizer selection")
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--do_summary', action='store_true', default=False, help='whether do summary or not')
    parser.add_argument('--pretrained_dir', required=False,type=str, help="the path of pretrained models")
    return parser.parse_args()
