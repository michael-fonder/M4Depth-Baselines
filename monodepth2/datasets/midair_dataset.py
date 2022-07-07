# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from torchvision import transforms
from PIL import Image  # using pillow-simd for increased speed
import scipy.misc as misc

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class MidAirDataset(MonoDataset):
    """Class for MidAir dataset loader
        """

    def __init__(self, *args, **kwargs):
        super(MidAirDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.5, 0, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1024, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        folder = line[0]
        frame_index = int(line[1])*4

        f_str = str(frame_index * 4).zfill(6) + "." + "PNG"
        folder = folder.replace("sensor", "stereo_disparity")

        depth_path = os.path.join(self.data_path, folder, f_str)

        return os.path.isfile(depth_path)

    def get_image_path(self, folder, frame_index, side):
        f_str = str(frame_index).zfill(6) + "." + "JPEG"
        if side == "l":
            folder = folder.replace("sensor", "color_left")
        else:
            folder = folder.replace("sensor", "color_right")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index*4, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = str(frame_index*4).zfill(6) + "." + "PNG"
        folder = folder.replace("sensor", "stereo_disparity")
        depth_path = os.path.join(
            self.data_path,
            folder,
            f_str)

        def open_float16(image_path, h, w):
            pic = pil.open(image_path)
            pic = pic.resize([h, w], pil.NEAREST)
            img = np.asarray(pic, np.uint16)
            img.dtype = np.float16
            return img

        disp_gt = open_float16(depth_path, self.height, self.width)
        depth_gt = 512.0/disp_gt.astype(np.float32)
        depth_gt = np.clip(depth_gt, 1., 100.)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
