import numpy as np
import cv2
import argparse
from evaluation_utils import *

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split',               type=str,   help='data split, kitti, midair, tartanair or eigen',         required=True)
parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--db_path',             type=str,   help='path to dataset',   required=True)
parser.add_argument('--min_depth',           type=float, help='minimum depth for evaluation',        default=1e-3)
parser.add_argument('--max_depth',           type=float, help='maximum depth for evaluation',        default=80)
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',   action='store_true')
parser.add_argument('--export_pics', dest="export_pics", action="store_true", help="Export maps in files")
parser.set_defaults(export_pics=False)

args = parser.parse_args()

if __name__ == '__main__':

    pred_disparities = np.load(args.predicted_disp_path)

    if args.split == 'midair':
        
        gt_disparities = load_gt_disp_midair(args.db_path, args.gt_path)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_midair(gt_disparities, pred_disparities)
        num_samples = len(gt_depths)

    elif 'tartanair' in args.split:
        print("use tartanair-split")
        gt_depths = load_gt_depth_tartanair(args.db_path, args.gt_path)
        pred_depths = [1./disp for disp in pred_disparities+0.000001]
        num_samples = len(gt_depths)

    elif args.split == 'eigen':
        num_samples = 697
        test_files = read_text_lines(args.gt_path + 'eigen_test_files.txt')
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.gt_path)

        num_test = len(im_files)
        gt_depths = []
        pred_depths = []
        for t_id in range(num_samples):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            gt_depths.append(depth.astype(np.float32))

            disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
            disp_pred = disp_pred * disp_pred.shape[1]

            # need to convert from disparity to depth
            focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
            depth_pred = (baseline * focal_length) / disp_pred
            depth_pred[np.isinf(depth_pred)] = 0

            pred_depths.append(depth_pred)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):
        
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        def save_maps(gt, est, id):

            def save_depth(map, path):
                map = np.clip(map, np.exp(0.), 80.)
                I8_content = (np.log(map) * 255.0 / np.log(80.)).astype(np.uint8)
                img_tmp = Image.fromarray(np.stack((I8_content, I8_content, I8_content), axis=2))
                img_tmp.save(path)

            save_dir = os.path.join(*["results", args.split, "saved_maps"])
            print(gt.shape)
            os.makedirs(save_dir, exist_ok=True)
            filename_gt = os.path.join(save_dir, str(id).zfill(6) + "_gt.png")
            save_depth(gt, filename_gt)
            filename_est = os.path.join(save_dir, str(id).zfill(6) + "_est.png")
            save_depth(est, filename_est)

        if args.export_pics:
            save_maps(gt_depth, pred_depth, i)
        
        mask = gt_depth > 0
        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        gt_depth[gt_depth < args.min_depth] = args.min_depth
        gt_depth[gt_depth > args.max_depth] = args.max_depth

        if args.split == 'eigen':
            mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

            
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape

                # crop used by Garg ECCV16
                # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                if args.garg_crop:
                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                # crop we found by trial and error to reproduce Eigen NIPS14 results
                elif args.eigen_crop:
                    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

        if args.split == 'kitti':
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        if args.split=="midair":
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth, pred_depth)
        else:
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            pred_depth *= ratio
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
