#!/usr/bin/env python
#
# Copyright M4Depth authors 2021. All rights reserved.
# ==============================================================================

import tensorflow as tf
from utils import dense_image_warp
import os

class Generic_Dataset():

    def __init__(self, db_path, db_seq_len=8, seq_len=4, batch_size=3, records=None, training=False):
        print("Path at terminal when executing this file")
        print(os.getcwd() + "\n")

        # Images and camera properties
        # self.optical_center = None # [cx,cy]
        # self.focal_length = None # [fx,fy]
        # self.in_size = None # [h,w]
        # self.out_size = None # [h,w]
        # self.records_path = "data"

        # Computed automatically
        self.out_focal_length = [self.focal_length[i]*self.out_size[-(i+1)]/self.in_size[-(i+1)] for i in range(2)]
        self.out_optical_center = [self.optical_center[i]*self.out_size[-(i+1)]/self.in_size[-(i+1)] for i in range(2)]

        if records is not None:
            self.records = records
        else:
            if training:
                self.records = 'train_data.tfrecords'
            else:
                self.records = 'test_data.tfrecords'

        # Class related properties
        self.batch_size=batch_size
        self.is_training=training
        self.db_seq_len = db_seq_len
        self.seq_len = seq_len

        self.db_path = db_path # "datasets/MidAir-RAM"

        self.new_traj = tf.convert_to_tensor([i==0 for i in range(seq_len)], dtype=tf.bool)
        self.data = None
        if training:
            with tf.name_scope("train"):
                self._build_train_dataset()
        else:
            with tf.name_scope("test"):
                self._build_eval_dataset()

        self.length = self.data.cardinality().numpy()
        self.depth_type = "map"

    def _build_train_dataset(self):
        dataset = tf.data\
            .TFRecordDataset(filenames=[os.path.join(self.records_path, self.records)])\
            .map(self._extract_sequences, num_parallel_calls=tf.data.AUTOTUNE)

        sample_count = 0
        for seq_data, seq_len in dataset:
            sample_count += seq_len // self.db_seq_len
            seq_data = tf.data.Dataset.from_tensor_slices(seq_data).batch(self.db_seq_len, drop_remainder=True)
            if self.data is None:
                self.data = seq_data
            else:
                self.data = self.data.concatenate(seq_data)
        # todo add cache
        self.data = self.data.shuffle(sample_count, reshuffle_each_iteration=True).unbatch().map(
            self._preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(self.db_seq_len, drop_remainder=True).map(
            self._data_augmentation, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size,
                                                                                drop_remainder=True).prefetch(50)


    def _build_eval_dataset(self):
        dataset = tf.data\
            .TFRecordDataset(filenames=[os.path.join(self.records_path, self.records)])\
            .map(self._extract_sequences, num_parallel_calls=tf.data.AUTOTUNE)

        for seq_data, seq_len in dataset:
            seq_data = tf.data.Dataset.from_tensor_slices(seq_data).map(self._preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
            if self.data is None:
                self.data = seq_data
            else:
                self.data = self.data.concatenate(seq_data)
        # todo add cache
        self.data = self.data.batch(1, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


    @tf.function
    def _extract_sequences(self, sequence_example):
        context = {"seq_len": tf.io.FixedLenFeature([], dtype=tf.int64)}
        seq_features = {"camera_l"  : tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                        "disp"      : tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                        "rot"       : tf.io.FixedLenSequenceFeature([3], dtype=tf.float32),
                        "trans"     : tf.io.FixedLenSequenceFeature([3], dtype=tf.float32),
                        "id"        : tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
                        }
        context_data, features = tf.io.parse_single_sequence_example(
            sequence_example, sequence_features=seq_features, context_features=context)

        return features, context_data['seq_len']

    @tf.function
    def _preprocess_sample(self, data_sample):
        out_data = data_sample.copy()
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_jpeg(file)
        out_data['RGB_im'] = tf.cast(image, dtype=tf.float32)/255.

        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['disp']], separator='/'))
        image = tf.image.decode_png(file, dtype=tf.uint16)
        image = tf.bitcast(image, tf.float16)
        out_data['depth'] = 512./tf.cast(image, dtype=tf.float32)

        out_data['file_name'] = data_sample['disp']

        if not self.is_training:
            out_data['RGB_im'] = tf.reshape(tf.image.resize(out_data['RGB_im'], self.out_size), self.out_size+[3])
            out_data['depth'] = tf.reshape(tf.image.resize(out_data['depth'], self.out_size), self.out_size+[1])
            out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)#tf.strings.regex_full_match(data_sample['camera_l'], self.first_seq_sample_identifier)
            camera_data = {
                "f": tf.convert_to_tensor(self.out_focal_length),
                "c": tf.convert_to_tensor(self.out_optical_center)
            }
            out_data["camera"] = camera_data.copy()

        return out_data

    def random_crop_and_resize_image(self, image, zoom_factor):
        with tf.name_scope('random_crop_and_resize'):
            h, w, c = image.get_shape().as_list()

            # Set random value
            local_patch_size = tf.divide([float(self.in_size[0]), float(self.in_size[1])], zoom_factor)
            local_patch_size = tf.cast([local_patch_size[0], local_patch_size[1]], dtype=tf.int32)
            # Crop and resize
            image = tf.image.random_crop(image, tf.concat([[local_patch_size[0]], [local_patch_size[1]], [c]], axis=0))
            image = tf.image.resize(image, self.out_size, tf.image.ResizeMethod.BILINEAR)
            return image

    @tf.function
    def _data_augmentation(self, data_batch):
        out_data = {}
        im_color = data_batch["RGB_im"]
        im_depth = data_batch["depth"]
        rot = data_batch["rot"]
        pos = data_batch["trans"]

        if not self.is_training:
            # self.out_size = self.in_size
            zoom_factor = tf.random.uniform([1], minval=1.0, maxval=1.0)
        else:
            zoom_factor = tf.random.uniform([1], minval=1.0, maxval=1.0)

        if self.is_training:
            offset = tf.random.uniform(shape=[], minval=0, maxval= self.db_seq_len-self.seq_len+1, dtype=tf.int32)
            im_color = tf.slice(im_color, [offset, 0, 0, 0], [self.seq_len]+self.in_size+[3])
            im_depth = tf.slice(im_depth, [offset, 0, 0, 0], [self.seq_len]+self.in_size+[1])
            rot = tf.slice(rot, [offset, 0], [self.seq_len,3])
            pos = tf.slice(pos, [offset, 0], [self.seq_len,3])

        else:
            print("test seq!")
            im_color = im_color[-self.seq_len:,:,:,:]
            im_depth = im_depth[-self.seq_len:,:,:,:]
            rot = rot[-self.seq_len:,:]
            pos = pos[-self.seq_len:,:]

        im_color = tf.concat(tf.unstack(im_color, axis=0), axis=-1)
        im_depth = tf.concat(tf.unstack(im_depth, axis=0), axis=-1)

        cropped_data = self.random_crop_and_resize_image(tf.concat([im_depth, im_color], axis=-1), zoom_factor)

        color_data = tf.split(cropped_data[:, :, self.seq_len:], self.seq_len, axis=-1)
        depth_data = tf.split(cropped_data[:, :, :self.seq_len], self.seq_len, axis=-1)
        im_color = tf.reshape(tf.stack(color_data, axis=0), [self.seq_len] + self.out_size + [3])
        im_depth = tf.reshape(tf.stack(depth_data, axis=0), [self.seq_len] + self.out_size + [1])

        if self.is_training:
            im_color = tf.image.random_brightness(im_color, 0.2)
            im_color = tf.image.random_contrast(im_color, 0.75, 1.25)
            im_color = tf.image.random_saturation(im_color, 0.75, 1.25)
            im_color = tf.image.random_hue(im_color, 0.4)

            def do_nothing():
                return [im_color, im_depth, rot, pos]

            def true_flip_v():
                col = tf.reverse(im_color, axis=[1])
                dep = tf.reverse(im_depth, axis=[1])
                r = tf.multiply(rot, [[-1.,1.,-1.]])
                t = tf.multiply(pos, [[1.,-1.,1.]])
                return [col, dep, r, t]

            p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            pred = tf.less(p_order, 0.5)
            im_color, im_depth, rot, pos = tf.cond(pred, true_flip_v, do_nothing)

            if self.out_size[0] == self.out_size[1]:
                def true_transpose():
                    col = tf.transpose(im_color, perm=[0,2,1,3])
                    dep = tf.transpose(im_depth, perm=[0,2,1,3])
                    r = tf.stack([-rot[:,1], -rot[:,0], -rot[:,2]], axis=1)
                    t = tf.stack([pos[:,1], pos[:,0], pos[:,2]], axis=1)
                    return [col, dep, r, t]

                p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(p_order, 0.5)
                im_color, im_depth, rot, pos = tf.cond(pred, true_transpose, do_nothing)

            def true_inv_col():
                return [1.-im_color, im_depth, rot, pos]

            p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            pred = tf.less(p_order, 0.5)
            im_color, im_depth, rot, pos = tf.cond(pred, true_inv_col, do_nothing)


        # out_data["focal_length"] = tf.ones([self.seq_len, 2])*self.out_size*zoom_factor/2.
        camera_data = {
            "f": tf.convert_to_tensor(self.out_focal_length)*zoom_factor,
            "c": tf.convert_to_tensor(self.out_optical_center)
        }
        out_data["camera"]  = camera_data.copy()
        out_data["depth"]   = tf.reshape(im_depth, [self.seq_len] + self.out_size + [1])
        out_data["RGB_im"]  = tf.reshape(im_color, [self.seq_len] + self.out_size + [3])
        out_data["rot"]     = rot
        out_data["trans"]   = pos
        out_data["new_traj"] = self.new_traj

        print(out_data)

        return out_data


class MidAir_Dataset(Generic_Dataset):

    def __init__(self, db_path, db_seq_len=8, seq_len=3, batch_size=3, records=None, training=False):
        # Images and camera properties
        self.optical_center = [512., 512.]  # [cx,cy]
        self.focal_length = [512., 512.]  # [fx,fy]
        self.in_size = [1024, 1024]  # [h,w]
        self.out_size = [384, 384]  # [h,w]

        self.records_path = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"data", "MidAir"])

        super(MidAir_Dataset, self).__init__(db_path, db_seq_len=db_seq_len, seq_len=seq_len, batch_size=batch_size, records=records, training=training)

class TartanAir_Dataset(Generic_Dataset):

    def __init__(self, db_path, db_seq_len=8, seq_len=3, batch_size=3, records=None, training=False):
        # Images and camera properties
        self.optical_center = [320., 240.]  # [cx,cy]
        self.focal_length = [320., 240.]  # [fx,fy]
        self.in_size = [480, 640]  # [h,w]
        self.out_size = [384, 512]  # [h,w]

        self.records_path = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"data", "TartanAir"])

        super(TartanAir_Dataset, self).__init__(db_path, db_seq_len=db_seq_len, seq_len=seq_len, batch_size=batch_size, records=records, training=training)

    @tf.function
    def _extract_sequences(self, sequence_example):
        context = {"seq_len": tf.io.FixedLenFeature([], dtype=tf.int64)}
        seq_features = {"camera_l": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                        "depth": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                        "rot": tf.io.FixedLenSequenceFeature([3], dtype=tf.float32),
                        "trans": tf.io.FixedLenSequenceFeature([3], dtype=tf.float32),
                        "intrinsics": tf.io.FixedLenSequenceFeature([4], dtype=tf.float32),
                        "id": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
                        }
        context_data, features = tf.io.parse_single_sequence_example(
            sequence_example, sequence_features=seq_features, context_features=context)

        return features, context_data['seq_len']

    @tf.function
    def _preprocess_sample(self, data_sample):
        out_data = data_sample.copy()
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_png(file)
        image = tf.cast(image, dtype=tf.float32)/255.
        out_data['RGB_im'] = tf.reshape(image, self.in_size+[3])
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['depth']], separator='/'))
        image = tf.io.decode_raw(file, tf.float32)
        image = image[-(self.in_size[0]*self.in_size[1]):]
        # tf.print(tf.strings.length(file))
        # tf.print(data_sample['depth'])
        # tf.print(image)
        out_data['depth'] = tf.reshape(tf.cast(image, dtype=tf.float32), self.in_size+[1])

        out_data['file_name'] = data_sample['depth']

        print(out_data)

        if not self.is_training:
            out_data['RGB_im'] = tf.reshape(tf.image.resize(out_data['RGB_im'], self.out_size), self.out_size+[3])
            out_data['depth'] = tf.reshape(tf.image.resize(out_data['depth'], self.out_size), self.out_size+[1])
            out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)#tf.strings.regex_full_match(data_sample['camera_l'], self.first_seq_sample_identifier)
            camera_data = {
                "f": tf.convert_to_tensor(self.out_focal_length),
                "c": tf.convert_to_tensor(self.out_optical_center)
            }
            out_data["camera"] = camera_data.copy()
        # tf.print(out_data['trans'])
        # tf.print(out_data['rot'])



        return out_data


class Kitti_Dataset(Generic_Dataset):

    def __init__(self, db_path, db_seq_len=8, seq_len=3, batch_size=3, records=None, training=False):
        # Images and camera properties
        self.optical_center = [607.1928, 185.2157]  # [cx,cy]
        self.focal_length = [718.856, 718.856]  # [fx,fy]
        self.in_size = [370, 1220]  # [h,w]
        self.out_size = [256, 768]  # [h,w] 256, 768

        # Computed automatically
        self.undistort_zoom = 1.
        self.out_focal_length = [self.undistort_zoom*self.focal_length[i]*self.out_size[-(i+1)]/self.in_size[-(i+1)] for i in range(2)]
        self.out_optical_center = [self.optical_center[i]*self.out_size[-(i+1)]/self.in_size[-(i+1)] for i in range(2)]

        self.records_path = os.path.join(*[os.path.dirname(os.path.realpath(__file__)),"data", "Kitti"])

        super(Kitti_Dataset, self).__init__(db_path, db_seq_len=db_seq_len, seq_len=seq_len, batch_size=batch_size, records=records, training=training)
        print(self.out_focal_length)

        print(self.records_path)
        self.depth_type = "velodyne"

    @tf.function
    def _extract_sequences(self, sequence_example):
        context = {"seq_len": tf.io.FixedLenFeature([], dtype=tf.int64)}
        seq_features = {"camera_l": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                        "velodyne": tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                        "rot": tf.io.FixedLenSequenceFeature([4], dtype=tf.float32),
                        "intrinsics": tf.io.FixedLenSequenceFeature([4], dtype=tf.float32),
                        "trans": tf.io.FixedLenSequenceFeature([3], dtype=tf.float32),
                        "id": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
                        }
        context_data, features = tf.io.parse_single_sequence_example(
            sequence_example, sequence_features=seq_features, context_features=context)

        return features, context_data['seq_len']

    @tf.function
    def _preprocess_sample(self, data_sample):
        out_data = data_sample.copy()
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_png(file)
        out_data['RGB_im'] = tf.cast(image, dtype=tf.float32) / 255.
        out_data['RGB_im'] = tf.reshape(tf.slice(out_data['RGB_im'], [0, 0, 0], self.in_size + [3]), self.in_size + [3])

        im_size_ratio = [self.out_size[1]/self.in_size[1], self.out_size[0]/self.in_size[0]]

        if 'velodyne' in data_sample:
            file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['velodyne']], separator='/'))
            image = tf.image.decode_png(file, dtype=tf.uint16)
            points = tf.cast(tf.bitcast(image, tf.float16), tf.float32)
            coords_x = tf.clip_by_value(points[:,1,0]*self.undistort_zoom*im_size_ratio[0], 0, self.out_size[1]-1.)
            coords_y = tf.clip_by_value(points[:,0,0]*self.undistort_zoom*im_size_ratio[1], 0, self.out_size[0]-1.)
            coords = tf.stack([coords_y, coords_x], axis=1)
            coords = tf.cast(coords, tf.int32)
            depths = points[:,-1,0]

            ones = tf.ones_like(depths)
            d_map = tf.scatter_nd(coords, depths, self.out_size)/(tf.scatter_nd(coords, ones, self.out_size)+1e-7)

            out_data['depth'] = tf.reshape(d_map, self.out_size+[1])


        if not self.is_training:
            out_data['RGB_im'] = tf.reshape(tf.image.resize(out_data['RGB_im'], self.out_size), self.out_size + [3])

            out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)
        camera_data = {
            "f": data_sample['intrinsics'][:2]*im_size_ratio,
            "c": data_sample['intrinsics'][2:]*im_size_ratio,
        }
        out_data["camera"] = camera_data.copy()

        return out_data

    @tf.function
    def _data_augmentation(self, data_batch):
        out_data = {}
        im_color = data_batch["RGB_im"]
        im_depth = data_batch["depth"]
        rot = data_batch["rot"]
        pos = data_batch["trans"]
        camera = data_batch["camera"]

        if not self.is_training:
            # self.out_size = self.in_size
            zoom_factor = tf.random.uniform([1], minval=1.0, maxval=1.0)
        else:
            zoom_factor = tf.random.uniform([1], minval=1.0, maxval=1.0)

        if self.is_training:
            offset = tf.random.uniform(shape=[], minval=0, maxval=self.db_seq_len - self.seq_len + 1, dtype=tf.int32)
            im_color = tf.slice(im_color, [offset, 0, 0, 0], [self.seq_len] + self.in_size + [3])
            im_depth = tf.slice(im_depth, [offset, 0, 0, 0], [self.seq_len] + self.out_size + [1])
            rot = tf.slice(rot, [offset, 0], [self.seq_len, 4])
            pos = tf.slice(pos, [offset, 0], [self.seq_len, 3])

        else:
            print("test seq!")
            im_color = im_color[-self.seq_len:, :, :, :]
            im_depth = im_depth[-self.seq_len:, :, :, :]
            rot = rot[-self.seq_len:, :]
            pos = pos[-self.seq_len:, :]

        im_color = tf.concat(tf.unstack(im_color, axis=0), axis=-1)
        # im_depth = tf.concat(tf.unstack(im_depth, axis=0), axis=-1)

        cropped_data = self.random_crop_and_resize_image(tf.concat([im_color], axis=-1), zoom_factor)

        color_data = tf.split(cropped_data, self.seq_len, axis=-1)
        im_color = tf.reshape(tf.stack(color_data, axis=0), [self.seq_len] + self.out_size + [3])

        if self.is_training:
            im_color = tf.image.random_brightness(im_color, 0.2)
            im_color = tf.image.random_contrast(im_color, 0.75, 1.25)
            im_color = tf.image.random_saturation(im_color, 0.75, 1.25)
            im_color = tf.image.random_hue(im_color, 0.4)

            def do_nothing():
                return [im_color, im_depth, rot, pos]

            def true_flip_v():
                col = tf.reverse(im_color, axis=[1])
                dep = tf.reverse(im_depth, axis=[1])
                r = tf.multiply(rot, [[-1., 1., -1.]])
                t = tf.multiply(pos, [[1., -1., 1.]])
                return [col, dep, r, t]

            # p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            # pred = tf.less(p_order, 0.5)
            # im_color, im_depth, rot, pos = tf.cond(pred, true_flip_v, do_nothing)

            def true_transpose():
                col = tf.transpose(im_color, perm=[0, 2, 1, 3])
                dep = tf.transpose(im_depth, perm=[0, 2, 1, 3])
                r = tf.stack([-rot[:, 1], -rot[:, 0], -rot[:, 2]], axis=1)
                t = tf.stack([pos[:, 1], pos[:, 0], pos[:, 2]], axis=1)
                return [col, dep, r, t]

            # p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            # pred = tf.less(p_order, 0.5)
            # im_color, im_depth, rot, pos = tf.cond(pred, true_transpose, do_nothing)

            def true_inv_col():
                return [1. - im_color, im_depth, rot, pos]

            # p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            # pred = tf.less(p_order, 0.5)
            # im_color, im_depth, rot, pos = tf.cond(pred, true_inv_col, do_nothing)

        # out_data["focal_length"] = tf.ones([self.seq_len, 2])*self.out_size*zoom_factor/2.
        camera_data = {
            "f": camera['f'][0,:]*zoom_factor, #tf.convert_to_tensor(self.out_focal_length) * zoom_factor,
            "c": camera['c'][0,:], #tf.convert_to_tensor(self.out_optical_center)
        }
        out_data["camera"] = camera_data.copy()
        out_data["depth"] = im_depth
        out_data["RGB_im"] = im_color
        out_data["rot"] = rot
        out_data["trans"] = pos
        out_data["new_traj"] = self.new_traj

        return out_data
