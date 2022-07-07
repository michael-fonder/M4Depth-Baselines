import json
import os.path
from dataloaders import DataloaderParameters

class PWCDCNetOptions:
    def __init__(self, args):

        # Global Options
        args.add_argument('--dataset',
                          default="",
                          choices=['midair', 'tartanair', 'kitti-raw'],
                          help="""Dataset to use (midair, tartanair or kitti-raw)""")
        args.add_argument('--db_path_config',
                          default=os.path.join(os.path.dirname(__file__),"datasets_location.json"),
                          help="""Json file with datasets path configuration""")
        args.add_argument('--ckpt_dir',
                          default="ckpt",
                          help="""Model checkpoint directory""")
        args.add_argument('-b', '--batch_size',
                          default=3, type=int,
                          help="""Size of each minibatch for each GPU""")
        args.add_argument('--mode',
                          choices=['train', 'eval', 'finetune', 'predict', 'validation'],
                          help="""Model run mode (train, finetune or eval)""")
        args.add_argument('--records_path',
                          default=None, type=str,
                          help="""csv files to use when loading dataset""")

        # Train Options
        args.add_argument('--data_aug',
                          default=False, action="store_true",
                          help="Perform data augmentation")
        args.add_argument('--db_seq_len',
                          default=None, type=int,
                          help="""Dataset sequence length (frames)""")
        args.add_argument('--seq_len',
                          default=4, type=int,
                          help="""Sequence length (frames)""")
        args.add_argument('--log_dir',
                          default=None,
                          help="""Tensorboard log directory""")
        args.add_argument('--summary_interval',
                          default=1200, type=int,
                          help="""How often (in batches) to update summaries.""")
        args.add_argument('--save_interval', default=2, type=int,
                          help="""How often (in epochs) to save checkpoints.""")
        args.add_argument('--enable_validation',
                          default=False, action="store_true",
                          help="Perform validation after each training epoch")
        args.add_argument('--no_augmentation',
                          default=False, action="store_true",
                          help="Disable data augmentation")

        args = args

        cmd, test_args = args.parse_known_args()
        json_data = json.load(open(cmd.db_path_config))

        path_root = os.path.dirname(__file__)
        for dataset, path in json_data.items():
            if not os.path.isabs(path):
                abs_path = os.path.join(path_root, path)
                json_data[dataset] = os.path.normpath(abs_path)

        self.dataloader_settings = DataloaderParameters(json_data, cmd.records_path, cmd.db_seq_len,
                                                        cmd.seq_len, not cmd.no_augmentation)