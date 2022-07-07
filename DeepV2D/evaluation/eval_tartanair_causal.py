import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
import argparse
import glob
import vis
import pickle

import eval_utils
from core import config
from deepv2d import DeepV2D
from data_stream.tartanair_causal import TartanAir

def process_for_evaluation(depth, scale):
    """ During training ground truth depths are scaled and cropped, we need to 
        undo this for evaluation """
    depth = (1.0/scale) * depth
    return depth

def make_predictions(args):
    """ Run inference over the test images """

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    db = TartanAir(args.dataset_dir, args.env)
    scale = db.args['scale']
 
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode='keyframe', est_pose=False)

    with tf.Session() as sess:
        deepv2d.set_session(sess)

        predictions = []
        groundtruth = []
        cnter = 0

        for (images, poses, intrinsics, depth_gt) in db.test_set_iterator():
            groundtruth.append(process_for_evaluation(depth_gt, scale))
            # print(depth_gt)
            # print(images[0])

            depth_predictions, _ = deepv2d(images, intrinsics, iters=args.n_iters, poses=poses)
        
            keyframe_depth = depth_predictions[0]
            keyframe_image = images[0]

            pred = process_for_evaluation(keyframe_depth, scale)
            predictions.append(pred.astype(np.float32))

            if args.viz:
                image_and_depth = vis.create_image_depth_figure(keyframe_image, keyframe_depth)
                cv2.imwrite('outputs/image%se.png' % str(cnter).zfill(3), image_and_depth)
                image_and_depth = vis.create_image_depth_figure(keyframe_image, depth_gt)
                cv2.imwrite('outputs/image%sg.png' % str(cnter).zfill(3), image_and_depth)
                cnter+=1
                # cv2.imshow('image', image_and_depth/255.0)
                # cv2.waitKey(10)

        return groundtruth, predictions


def evaluate(groundtruth, predictions, min_depth=1e-3, max_depth=80):
    depth_results = {}
    for depth_gt,depth_pr in zip(groundtruth,predictions):
        ht, wd = depth_gt.shape[:2]

        mask = depth_gt > 0
        depth_gt = np.clip(depth_gt, min_depth, max_depth)

        depth_pr = cv2.resize(depth_pr, (wd, ht))
        depth_pr = np.clip(depth_pr, min_depth, max_depth)

        depth_gt = depth_gt[mask]
        depth_pr = depth_pr[mask]

        depth_pr = np.median(depth_gt/depth_pr) * depth_pr
        depth_metrics = eval_utils.compute_depth_errors(
            depth_gt, depth_pr, min_depth=min_depth, max_depth=max_depth)

        if len(depth_results) == 0:
            for dkey in depth_metrics:
                depth_results[dkey] = []

        for dkey in depth_metrics:
            depth_results[dkey].append(depth_metrics[dkey])


    # aggregate results
    for dkey in depth_results:
        depth_results[dkey] = np.mean(depth_results[dkey])

    print(("{:>10}, "*len(depth_results)).format(*depth_results.keys()))
    print(("{:10.4f}, "*len(depth_results)).format(*depth_results.values()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/tartanair.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/kitti.ckpt', help='path to model checkpoint')
    parser.add_argument('--env', default="", choices=['urban', 'unstructured'], help="""Environement to test on ('urban' or 'unstructured')""")
    parser.add_argument('--dataset_dir', default='data/kitti/raw', help='config file used to train the model')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    parser.add_argument('--n_iters', type=int, default=5, help='number of video frames to use for reconstruction')
    args = parser.parse_args()

    # run inference on the test images
    groundtruth, predictions = make_predictions(args)

    # evaluate on KITTI test set
    evaluate(groundtruth, predictions)