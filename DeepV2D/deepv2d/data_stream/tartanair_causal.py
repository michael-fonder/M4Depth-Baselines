import tensorflow as tf
import numpy as np
from data_stream import util

import csv
import cv2
import os
import time
import random
import glob


class TartanAir(object):
    default_args = {
        'frames': 4,
        'hist_len': 4,
        # 'height': 192,
        # 'width': 1088,
        'height': 384,
        'width': 512,
        'scale': 0.05,
    }

    def __init__(self, dataset_path, env, mode='train', args=default_args):

        self.dataset_path = dataset_path
        self.args = args
        self.mode = mode
        self.test_list = []
        self.hist_len = 4

        csv_files = glob.glob(os.path.join(*['data/tartanair', env, "**/*.csv"]), recursive=True)
        for file_name in csv_files:
            with open(file_name, 'r') as f:
                traj_list = []
                reader = csv.reader(f, delimiter='\t')
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    traj_list.append(row[1:])
                self.test_list.append(traj_list.copy())

        print("%i samples found for tartanair %s" % (len(self.test_list), env))

    def _load_example(self, sequence):

        n_frames = len(sequence)

        center_idx = len(sequence)-1
        # put the keyframe at the first index
        sequence = [sequence[center_idx]] + \
            [sequence[i] for i in range(n_frames) if not i==center_idx]

        images, poses = [], []
        self.ref_pose = None
        for frame in sequence:
            img = self._load_image(frame['image'])
            images.append(img)
            pose = self._get_pose(frame['quat'], frame['trans'])
            poses.append(pose)

        depth = self._load_depth(sequence[0]['depth'])
        depth = cv2.resize(depth, (self.args['width'], self.args['height']))


        intrinsics = self._load_intrinsics().astype("float32")
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (self.args['width'], self.args['height']))


        # WARNING pixels with no color information are disabled in the depth map
        greyscale_frame = np.linalg.norm(images[0], ord=2, axis=-1)
        depth[greyscale_frame == 0] = 0

        images = np.array(images, dtype="float32")
        depth = depth.astype("float32")

        example_blob = {
            'images': images,
            'poses': poses,
            'depth': depth,
            'intrinsics': intrinsics,
        }

        return example_blob

    def test_set_iterator(self, ):
        for traj in self.test_list:
            for index, _ in enumerate(traj[self.hist_len:]):
                frame = index+self.hist_len
                seq = []
                for j in range(frame-self.hist_len, frame+1):
                    frame = {
                        'image': traj[j][0],
                        'depth': traj[j][1],
                        'quat' : [float(nbre) for nbre in traj[j][2:6]],
                        'trans': [float(nbre) for nbre in traj[j][6:]]
                    }
                    seq.append(frame)

                data_blob = self._load_example(seq)
                yield data_blob['images'], data_blob['poses'], data_blob['intrinsics'], data_blob['depth']



    def _load_intrinsics(self):
        fx = 0.5 * self.args['width']
        fy = 2./3. * self.args['height']
        cx = 0.5 * self.args['width']
        cy = 0.5 * self.args['height']

        intrinsics = np.array([fx, fy, cx, cy])
        return intrinsics


    def _load_image(self, im_name):
        return cv2.imread(os.path.join(self.dataset_path, im_name))


    def _load_depth(self, depth_name):
        return np.load(os.path.join(self.dataset_path, depth_name)) * self.args['scale']

    def _get_pose(self, quaternion, translation):
        w, x, y, z = quaternion
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        rot_mat = np.array([[1.0 - (tyy + tzz), txy - twz, txz + twy],
                           [txy + twz, 1.0 - (txx + tzz), tyz - twx],
                           [txz - twy, tyz + twx, 1.0 - (txx + tyy)]])

        pose_w_cam = np.eye(4)
        pose_w_cam[:3,:3] = rot_mat
        pose_w_cam[:3,3] = np.array(translation) * self.args['scale']
        # if self.ref_pose is None:
        #     self.ref_pose = util.inv_SE3(pose)
        # pose = self.ref_pose @ pose
        pose_cam_w = util.inv_SE3(pose_w_cam) # needed to be coherent with other datasets
        return pose_cam_w
