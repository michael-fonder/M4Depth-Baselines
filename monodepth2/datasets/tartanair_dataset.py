# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
# Modifications brought by M4depth Authors 2021

from __future__ import absolute_import, division, print_function

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil
from torchvision import transforms
from PIL import Image  # using pillow-simd for increased speed
import scipy.misc as misc

from .mono_dataset import MonoDataset
import csv

class TartanAirDataset(MonoDataset):
    """Class for MidAir dataset loader
        """

    def __init__(self, *args, **kwargs):
        self.frameskip = int(2)
        super(TartanAirDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.5, 0, 0.5, 0],
                           [0, 0.667, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (480, 640)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.pose_files = {}
        # self.frame_idxs = [frame_idx * self.frameskip for frame_idx in self.frame_idxs]

    def __getitem__(self, index):
        inputs = super(TartanAirDataset, self).__getitem__(index)
        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
        for i in self.frame_idxs:
            inputs[("poses", i)] = self.get_pose(folder, frame_index + i)
        return inputs

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def check_depth(self):
        line = self.filenames[0].split()
        folder = line[0]
        frame_index = int(line[1])
        frame_index *= self.frameskip

        f_str = str(frame_index).zfill(6) + "_left_depth.npy"
        depth_path = os.path.join(*[self.data_path, folder, "depth_left", f_str])

        # depth_path = os.path.join(self.data_path, folder, f_str)

        return os.path.isfile(depth_path)

    def get_image_path(self, folder, frame_index, side):
        
        if side == "l":
            f_str = str(frame_index).zfill(6) + "_left.png"
            image_path = os.path.join(*[self.data_path, folder, "image_left", f_str])
        else:
            f_str = str(frame_index).zfill(6) + "_right.png"
            image_path = os.path.join(*[self.data_path, folder, "image_right", f_str])
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index*self.frameskip, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        frame_index *= self.frameskip
        f_str = str(frame_index).zfill(6) + "_left_depth.npy"
        path = os.path.join(*[self.data_path, folder, "depth_left", f_str])
        
        # WARNING this nullifies depth in areas with no color information 
        color = self.loader(self.get_image_path(folder, frame_index, side))
        greyscale = np.linalg.norm(color, axis=-1)
        mask = np.where(greyscale > 0, 1., 0.)

        depth_gt = np.load(path)
        depth_gt = np.clip(depth_gt, 1., 200.) * mask

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def _get_pose_mat(self, line):
        y, z, x, w = line[-4:] # ned to cam referential
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

        pose = np.eye(4)
        pose[:3,:3] = rot_mat
        pose[:3,3] = np.array([line[1],line[2],line[0]], dtype=np.float32)/22.
        return pose

    def get_pose(self, folder, frame_index):
        frame_index *= self.frameskip
        if not folder in self.pose_files:
            self.pose_files[folder] = []
            with open(os.path.join(*[self.data_path, folder, "pose_left.txt"]), 'r') as pose_file:
                reader = csv.reader(pose_file, delimiter=' ')
                for line in reader:
                    line = [float(item) for item in line]
                    self.pose_files[folder].append(self._get_pose_mat(line))
        return self.pose_files[folder][frame_index]