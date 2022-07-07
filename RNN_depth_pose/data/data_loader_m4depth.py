from __future__ import division
import tensorflow as tf
import random
import os, glob
from .protobuf_db import ProtoBufDeserializer
import math
import pandas as pd


class GenericDataLoader(object):
    """ Data loading class for training heatmap-attention-padding network

    Args:
        dataset_dir: Folder contain .tfrecords files
        batch_size: training batch size
        image_height, image_width: input image height and width
        opt: flags from input parser
    
    Returns:
        new_mask: A gauss smoothed tensor

    """

    def __init__(self,
                 dataset_dir,
                 records_dir,
                 batch_size,
                 image_height,
                 image_width,
                 num_epochs,
                 num_views):
        self.dataset_dir=dataset_dir
        self.batch_size=batch_size
        self.image_height=image_height
        self.image_width=image_width
        self.num_epochs = num_epochs
        self.num_views = num_views
        self.records_path = records_dir

        self.out_size = [image_height, image_width]

        self.db_seq_len = 8
        self.seq_len = num_views


    def _build_dataset(self):
        csv_files = glob.glob(os.path.join(*[self.records_path, "**/*.csv"]), recursive=True)

        sample_count = 0
        dataset=None
        for file in csv_files:
            pd_dataset = pd.read_csv(file, sep="\t")
            traj_dataset = tf.data.Dataset.from_tensor_slices(dict(pd_dataset))\
                .batch(self.db_seq_len, drop_remainder=True)
            # sample_count += seq_len // self.db_seq_len
            # seq_data = tf.data.Dataset.from_tensor_slices(seq_data).batch(self.db_seq_len, drop_remainder=True)
            if dataset is None:
                dataset = traj_dataset
            else:
                dataset = dataset.concatenate(traj_dataset)

        if dataset is None:
            raise Exception("No csv files found at the given path: %s" % self.records_path)

        if self.is_training:
            dataset = dataset.shuffle(100*self.batch_size, reshuffle_each_iteration=True)

        dataset = dataset.unbatch()\
            .map(self._decode_samples, num_parallel_calls=16)\
            .batch(self.db_seq_len, drop_remainder=True)\
            .map(self._build_sequence_samples, num_parallel_calls=16)

        return dataset

    def _decode_samples(self, data_sample):
        return NotImplementedError

    def _build_sequence_samples(self, data_sample):
        im_color = data_sample["RGB_im"]
        im_depth = data_sample["depth"]

        if self.is_training:
            offset = tf.random.uniform(shape=[], minval=0, maxval= self.db_seq_len-self.seq_len+1, dtype=tf.int32)
            im_color = tf.slice(im_color, [offset, 0, 0, 0], [self.seq_len]+self.out_size+[3])
            im_depth = tf.slice(im_depth, [offset, 0, 0, 0], [self.seq_len]+self.out_size+[1])

        out_data = {}
        out_data["image_seq"] = tf.concat(tf.unstack(im_color, axis=0), axis=1)
        out_data["depth_seq"] = tf.concat(tf.unstack(im_depth, axis=0), axis=1)
        out_data['intrinsics'] = tf.convert_to_tensor([[self.fx_rel*self.out_size[1],0.,self.cx_rel*self.out_size[1]],
                                                                    [0., self.fy_rel*self.out_size[0],self.cy_rel*self.out_size[0]],
                                                                    [0., 0., 1.]], dtype=tf.float32)
        # For testing, turn with_aug to false
        if self.is_training:
            out_data = self.data_augmentation2(out_data, self.image_height, self.image_width)

        return out_data

    #==================================
    # Load training data from tf records
    #==================================
    def inputs(self, is_training=True):
        """Reads input data num_epochs times.
        Args:
            batch_size: Number of examples per returned batch.
            num_epochs: Number of times to read the input data, or 0/None to
            train forever.
        Returns:
            data_dict: A dictional contain input image and groundtruth label
        """
        self.is_training = is_training
        #If epochs number is none, then infinite repeat
        if not self.num_epochs:
            self.num_epochs = None
        
        with tf.name_scope('input'):
            dataset = self._build_dataset()
            # The map transformation takes a function and applies it to every element
            # of the dataset.
            if is_training and False: # shuffle performed in _build_dataset
                dataset = dataset.shuffle(100*self.batch_size) # when testing we do not shuffle data
            dataset = dataset.repeat(self.num_epochs)        # Number of epochs to train
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(1)
            iterator = tf.data.make_one_shot_iterator(dataset)

        return iterator#.get_next()


    def data_augmentation2(self, data_dict, out_h, out_w, is_training = True):

        def flip_intrinsics(intrinsics, width):
            fx = intrinsics[0,0]
            fy = intrinsics[1,1]
            cx = width-intrinsics[0,2]
            cy = intrinsics[1,2]
            
            zeros = tf.zeros_like(fx)
            r1 = tf.stack([fx, zeros, cx])
            r2 = tf.stack([zeros, fy, cy])
            r3 = tf.constant([0.,0.,1.])
            intrinsics = tf.stack([r1, r2, r3], axis=0)  

            return intrinsics

        def flip_left_right(image_seq, num_views):
            """Perform random distortions on an image.
            Args:
            image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
            thread_id: Preprocessing thread id used to select the ordering of color
              distortions. There should be a multiple of 2 preprocessing threads.
            Returns:
            distorted_image: A float32 Tensor of shape [height, width, 3] with values in
              [0, 1].
            """

            in_h, in_w, _ = image_seq.get_shape().as_list()
            in_w = in_w/num_views
            # Randomly flip horizontally.
                
            for i in range(num_views):
                # Scale up images
                image = tf.slice(image_seq,
                                 [0, int(in_w) * i, 0],
                                 [-1, int(in_w), -1])

                image = tf.image.flip_left_right(image)
    
    
                if i == 0:
                    flip_image = image
                else:
                    flip_image = tf.concat([flip_image, image],axis=1)
    
            
            return flip_image


        
        # Random scaling
        def random_scaling(data_dict, num_views):
        
            in_h, in_w, _ = data_dict['image_seq'].get_shape().as_list()
            
            in_w = in_w/num_views                
            
            scaling = tf.random_uniform([1], 1, 1.25)
            x_scaling = scaling[0]
            y_scaling = scaling[0]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            
            scaled_images = []
            scaled_depths = []
            scaled_images_norm = []
            
            for i in range(num_views):
            
                # Scale up images
                image = tf.slice(data_dict['image_seq'],
                                 [0, int(in_w) * i, 0],
                                 [-1, int(in_w), -1])
                image = tf.image.resize_images(image, [out_h, out_w])
                scaled_images.append(image)
                
                
                # # Scale up normalized images
                # image_norm = tf.slice(data_dict['image_seq_norm'],
                #                  [0, int(in_w) * i, 0],
                #                  [-1, int(in_w), -1])
                # image_norm = tf.image.resize_images(image_norm, [out_h, out_w])
                # scaled_images_norm.append(image_norm)
                
                
                # Scale up depth
                depth = tf.slice(data_dict['depth_seq'],
                                 [0, int(in_w) * i, 0],
                                 [-1, int(in_w), -1])
                depth = tf.image.resize_images(depth, [out_h, out_w],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                scaled_depths.append(depth)
            
            return scaled_images, scaled_depths, scaled_images_norm



        # Random cropping
        def random_cropping(data_dict, scaled_images, scaled_depths, scaled_images_norm, num_views, out_h, out_w):

            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            in_h, in_w, _ = tf.unstack(tf.shape(scaled_images[0]))
            
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]

            _in_h = tf.to_float(in_h)
            _in_w = tf.to_float(in_w)
            _out_h = tf.to_float(out_h)
            _out_w = tf.to_float(out_w)

            
            
            fx = data_dict['intrinsics'][0,0]*_in_w/_out_w
            fy = data_dict['intrinsics'][1,1]*_in_h/_out_h
            cx = data_dict['intrinsics'][0,2]*_in_w/_out_w-tf.cast(offset_x, tf.float32)
            cy = data_dict['intrinsics'][1,2]*_in_h/_out_h-tf.cast(offset_y, tf.float32)
            
            zeros = tf.zeros_like(fx)
            r1 = tf.stack([fx, zeros, cx])
            r2 = tf.stack([zeros, fy, cy])
            r3 = tf.constant([0.,0.,1.])
            data_dict['intrinsics'] = tf.stack([r1, r2, r3], axis=0)            

            for i in range(num_views):
            
                if i == 0:
                    cropped_images =  tf.image.crop_to_bounding_box(
                                        scaled_images[i], offset_y, offset_x, out_h, out_w)
                    cropped_depths =  tf.image.crop_to_bounding_box(
                                        scaled_depths[i], offset_y, offset_x, out_h, out_w)
                    # cropped_images_norm =  tf.image.crop_to_bounding_box(
                    #                     scaled_images_norm[i], offset_y, offset_x, out_h, out_w)
                else:
                    cropped_images = tf.concat([cropped_images, tf.image.crop_to_bounding_box(
                                        scaled_images[i], offset_y, offset_x, out_h, out_w)], axis=1)
                    cropped_depths = tf.concat([cropped_depths, tf.image.crop_to_bounding_box(
                                        scaled_depths[i], offset_y, offset_x, out_h, out_w)], axis=1)
                    # cropped_images_norm = tf.concat([cropped_images_norm, tf.image.crop_to_bounding_box(
                    #                     scaled_images_norm[i], offset_y, offset_x, out_h, out_w)], axis=1)
            data_dict['image_seq'] = cropped_images    
            data_dict['depth_seq'] = cropped_depths 
            # data_dict['image_seq_norm'] = cropped_images_norm
            
            return data_dict


        #data_dict=random_rotate(data_dict)
        # do_augment  = tf.random_uniform([], 0, 1)
        # data_dict['image_seq'] = tf.cond(do_augment > 0.5, lambda: flip_left_right(data_dict['image_seq'],self.num_views), lambda: data_dict['image_seq'])
        # data_dict['depth_seq'] = tf.cond(do_augment > 0.5, lambda: flip_left_right(data_dict['depth_seq'],self.num_views), lambda: data_dict['depth_seq'])
        # data_dict['intrinsics'] = tf.cond(do_augment > 0.5, lambda: flip_intrinsics(data_dict['intrinsics'],out_w), lambda: data_dict['intrinsics'])

        scaled_images, scaled_depths, scaled_images_norm = random_scaling(data_dict, self.num_views)
        data_dict = random_cropping(data_dict, scaled_images, scaled_depths, scaled_images_norm, self.num_views, out_h, out_w)

        return data_dict


class MidAirDataLoader(GenericDataLoader):
    """ Data loading class for training heatmap-attention-padding network

    Args:
        dataset_dir: Folder contain .tfrecords files
        batch_size: training batch size
        image_height, image_width: input image height and width
        opt: flags from input parser

    Returns:
        new_mask: A gauss smoothed tensor

    """

    def __init__(self, *args, **kwargs):

        super(MidAirDataLoader, self).__init__(*args, **kwargs)

        self.in_size = [1024, 1024]
        self.fx_rel = 0.5
        self.fy_rel = 0.5
        self.cx_rel = 0.5
        self.cy_rel = 0.5

    def _decode_samples(self, data_sample):
        file = tf.io.read_file(tf.strings.join([self.dataset_dir, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_jpeg(file)
        rgb_image = tf.cast(image, dtype=tf.float32)/255.

        file = tf.io.read_file(tf.strings.join([self.dataset_dir, data_sample['disp']], separator='/'))
        image = tf.image.decode_png(file, dtype=tf.uint16)
        image = tf.bitcast(image, tf.float16)
        depth = (512./tf.cast(image, dtype=tf.float32))


        out_data = {}
        out_data['RGB_im'] = tf.reshape(tf.image.resize(rgb_image, self.out_size), self.out_size+[3])
        out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size), self.out_size+[1])
        return out_data

class TartanAirDataLoader(GenericDataLoader):
    """ Data loading class for training heatmap-attention-padding network

    Args:
        dataset_dir: Folder contain .tfrecords files
        batch_size: training batch size
        image_height, image_width: input image height and width
        opt: flags from input parser

    Returns:
        new_mask: A gauss smoothed tensor

    """

    def __init__(self, *args, **kwargs):

        super(TartanAirDataLoader, self).__init__(*args, **kwargs)

        self.in_size = [480, 640]
        self.fx_rel = 0.5
        self.fy_rel = 2./3.
        self.cx_rel = 0.5
        self.cy_rel = 0.5

    def _decode_samples(self, data_sample):
        file = tf.io.read_file(tf.strings.join([self.dataset_dir, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_jpeg(file)
        rgb_image = tf.cast(image, dtype=tf.float32)/255.

        im_greyscale = tf.math.reduce_euclidean_norm(rgb_image, axis=-1, keepdims=True)
        validity_mask = tf.cast(tf.greater(im_greyscale, 0.), tf.float32)

        file = tf.io.read_file(tf.strings.join([self.dataset_dir, data_sample['depth']], separator='/'))
        image = tf.io.decode_raw(file, tf.float32)
        image = image[-(self.in_size[0]*self.in_size[1]):]
        depth = tf.reshape(tf.cast(image, dtype=tf.float32), self.in_size+[1])*validity_mask

        out_data = {}
        out_data['RGB_im'] = tf.reshape(tf.image.resize(rgb_image, self.out_size), self.out_size+[3])
        out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size), self.out_size+[1])
        return out_data