import argparse

def _get_test_opt():
    parser = argparse.ArgumentParser(description = 'Evaluate performance of SARPN on NYU-D v2 test set')
    parser.add_argument('--testlist_path', help='the path of testlist')
    parser.add_argument('--test_files', help='the path of testfiles')
    parser.add_argument('--dataset', choices=['midair', 'tartanair'], required=True, help='dataset to use for the test')
    parser.add_argument('--root_path', required=True, help="the root path of dataset")
    parser.add_argument("--h", type=int, default=384, help="Desired output height")
    parser.add_argument("--w", type=int, default=384, help="Desired output width")
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--seq_len', type=int, default=5, help='testing sequence length')
    parser.add_argument('--loadckpt', required=True, help="the path of the loaded model")
    parser.add_argument('--export_pics', dest="export_pics", action="store_true", help="Export maps in files")
    parser.set_defaults(export_pics=False)
    # parse arguments
    return parser.parse_args()
