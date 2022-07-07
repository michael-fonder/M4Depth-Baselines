'''
PWCDCNet code borrowed from https://github.com/hellochick/PWCNet-tf2/blob/master/PWCDCNet.py
published under the MIT License - Copyright (c) 2022 Hsuan-Kung Yang
'''

import tensorflow as tf
from tensorflow import keras as ks
from utils.depth_operations import *
from tensorflow.python.ops import variables
from flow_utils import bilinear_warp
from tensorflow.keras import layers

IS_TRAINING =False


class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides, name=None, padding=1, dilation_rate=1):
        super(Conv2D, self).__init__(name=name)

        self.conv_out = layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding='same',
                                      kernel_initializer='he_normal',
                                      dilation_rate=dilation_rate,
                                      activation=layers.LeakyReLU(0.1))

    def call(self, inputs):
        x = self.conv_out(inputs)

        return x


class DeConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, name=None):
        super(DeConv2D, self).__init__(name=name)

        self.deconv_out = layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding='same',
                                                 name=name)

    def call(self, inputs):
        x = self.deconv_out(inputs)

        return x


'''
Function for calculate cost volume is borrowed from 
        - https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/core_costvol.py
        Copyright (c) 2018 Phil Ferriere
        MIT License

        which based on 
        - https://github.com/tensorpack/tensorpack/blob/master/examples/OpticalFlow/flownet_models.py

        Written by Patrick Wieschollek, Copyright Yuxin Wu
        Apache License 2.0
'''


def CostVolumn(c1, warp, search_range, name='cost_volumn'):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return cost_vol


class PredictFlow(layers.Layer):
    def __init__(self, name=None):
        super(PredictFlow, self).__init__()

        self.conv_out = layers.Conv2D(filters=2,
                                      kernel_size=3,
                                      strides=1,
                                      name=name,
                                      padding='same')

    def call(self, inputs):
        return self.conv_out(inputs)


class PWCDCNet(layers.Layer):
    '''
    Modified and inherited from the official pytorch version: https://github.com/NVlabs/PWC-Net/tree/master/PyTorch
    '''

    def __init__(self, max_displacement=4, is_training=False):
        self.is_training = is_training
        super(PWCDCNet, self).__init__()

        self.conv1a = Conv2D(16, kernel_size=3, strides=2, name='conv1a')
        self.conv1aa = Conv2D(16, kernel_size=3, strides=1, name='conv1aa')
        self.conv1b = Conv2D(16, kernel_size=3, strides=1, name='conv1b')
        self.conv2a = Conv2D(32, kernel_size=3, strides=2, name='conv2a')
        self.conv2aa = Conv2D(32, kernel_size=3, strides=1, name='conv2aa')
        self.conv2b = Conv2D(32, kernel_size=3, strides=1, name='conv2b')
        self.conv3a = Conv2D(64, kernel_size=3, strides=2, name='conv3a')
        self.conv3aa = Conv2D(64, kernel_size=3, strides=1, name='conv3aa')
        self.conv3b = Conv2D(64, kernel_size=3, strides=1, name='conv3b')
        self.conv4a = Conv2D(96, kernel_size=3, strides=2, name='conv4a')
        self.conv4aa = Conv2D(96, kernel_size=3, strides=1, name='conv4aa')
        self.conv4b = Conv2D(96, kernel_size=3, strides=1, name='conv4b')
        self.conv5a = Conv2D(128, kernel_size=3, strides=2, name='conv5a')
        self.conv5aa = Conv2D(128, kernel_size=3, strides=1, name='conv5aa')
        self.conv5b = Conv2D(128, kernel_size=3, strides=1, name='conv5b')
        self.conv6aa = Conv2D(196, kernel_size=3, strides=2, name='conv6aa')
        self.conv6a = Conv2D(196, kernel_size=3, strides=1, name='conv6a')
        self.conv6b = Conv2D(196, kernel_size=3, strides=1, name='conv6b')

        self.LeakyReLU = layers.LeakyReLU(0.1)

        self.conv6_0 = Conv2D(128, kernel_size=3, strides=1, name='conv6_0')
        self.conv6_1 = Conv2D(128, kernel_size=3, strides=1, name='conv6_1')
        self.conv6_2 = Conv2D(96, kernel_size=3, strides=1, name='conv6_2')
        self.conv6_3 = Conv2D(64, kernel_size=3, strides=1, name='conv6_3')
        self.conv6_4 = Conv2D(32, kernel_size=3, strides=1, name='conv6_4')
        self.deconv6 = DeConv2D(2, kernel_size=4, strides=2, name='deconv_6')
        self.upfeat6 = DeConv2D(2, kernel_size=4, strides=2, name='upfeat_6')

        self.conv5_0 = Conv2D(128, kernel_size=3, strides=1, name='conv5_0')
        self.conv5_1 = Conv2D(128, kernel_size=3, strides=1, name='conv5_1')
        self.conv5_2 = Conv2D(96, kernel_size=3, strides=1, name='conv5_2')
        self.conv5_3 = Conv2D(64, kernel_size=3, strides=1, name='conv5_3')
        self.conv5_4 = Conv2D(32, kernel_size=3, strides=1, name='conv5_4')
        self.deconv5 = DeConv2D(2, kernel_size=4, strides=2, name='deconv_5')
        self.upfeat5 = DeConv2D(2, kernel_size=4, strides=2, name='upfeat_5')

        self.conv4_0 = Conv2D(128, kernel_size=3, strides=1)
        self.conv4_1 = Conv2D(128, kernel_size=3, strides=1)
        self.conv4_2 = Conv2D(96, kernel_size=3, strides=1)
        self.conv4_3 = Conv2D(64, kernel_size=3, strides=1)
        self.conv4_4 = Conv2D(32, kernel_size=3, strides=1)
        self.deconv4 = DeConv2D(2, kernel_size=4, strides=2)
        self.upfeat4 = DeConv2D(2, kernel_size=4, strides=2)

        self.conv3_0 = Conv2D(128, kernel_size=3, strides=1)
        self.conv3_1 = Conv2D(128, kernel_size=3, strides=1)
        self.conv3_2 = Conv2D(96, kernel_size=3, strides=1)
        self.conv3_3 = Conv2D(64, kernel_size=3, strides=1)
        self.conv3_4 = Conv2D(32, kernel_size=3, strides=1)
        self.deconv3 = DeConv2D(2, kernel_size=4, strides=2)
        self.upfeat3 = DeConv2D(2, kernel_size=4, strides=2)

        self.conv2_0 = Conv2D(128, kernel_size=3, strides=1)
        self.conv2_1 = Conv2D(128, kernel_size=3, strides=1)
        self.conv2_2 = Conv2D(96, kernel_size=3, strides=1)
        self.conv2_3 = Conv2D(64, kernel_size=3, strides=1)
        self.conv2_4 = Conv2D(32, kernel_size=3, strides=1)
        self.deconv2 = DeConv2D(2, kernel_size=4, strides=2)

        self.dc_conv1 = Conv2D(128, kernel_size=3, strides=1, padding=1, dilation_rate=1)
        self.dc_conv2 = Conv2D(128, kernel_size=3, strides=1, padding=2, dilation_rate=2)
        self.dc_conv3 = Conv2D(128, kernel_size=3, strides=1, padding=4, dilation_rate=4)
        self.dc_conv4 = Conv2D(96, kernel_size=3, strides=1, padding=8, dilation_rate=8)
        self.dc_conv5 = Conv2D(64, kernel_size=3, strides=1, padding=16, dilation_rate=16)
        self.dc_conv6 = Conv2D(32, kernel_size=3, strides=1, padding=1, dilation_rate=1)

        self.predict_flow6 = PredictFlow(name='flow6_out')
        self.predict_flow5 = PredictFlow(name='flow5_out')
        self.predict_flow4 = PredictFlow(name='flow4_out')
        self.predict_flow3 = PredictFlow(name='flow3_out')
        self.predict_flow2 = PredictFlow(name='flow2_out')
        self.dc_conv7 = PredictFlow()

    def build(self, input_shape):

        if (not self.is_training):
            self.prev_frame = self.add_weight(name="prev_frame", shape=input_shape[:3] + [3], dtype='float32',
                                              initializer=tf.zeros_initializer(), trainable=False, use_resource=False)
        else:
            print("Skipping memory creation")

    def call(self, inputs, is_training=True):
        if (not self.is_training):
            im1 = inputs
            im2 = self.prev_frame
            # tf.print("im1")
            # tf.print(im1)
            # tf.print(im2)
        else:
            im1 = inputs[:, :, :, :3]
            im2 = inputs[:, :, :, 3:]

        print(im1)
        # im1 = inputs[:, :, :, :3]
        # im2 = inputs[:, :, :, 3:]

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        ### 6th flow
        corr6 = CostVolumn(c1=c16, warp=c26, search_range=4)
        x = tf.concat([self.conv6_0(corr6), corr6], 3)
        x = tf.concat([self.conv6_1(x), x], 3)
        x = tf.concat([self.conv6_2(x), x], 3)
        x = tf.concat([self.conv6_3(x), x], 3)
        x = tf.concat([self.conv6_4(x), x], 3)

        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        ### 5th flow
        warp5 = bilinear_warp(c25, up_flow6 * 0.625)
        corr5 = CostVolumn(c1=c15, warp=warp5, search_range=4)

        x = tf.concat([corr5, c15, up_flow6, up_feat6], 3)
        x = tf.concat([self.conv5_0(x), x], 3)
        x = tf.concat([self.conv5_1(x), x], 3)
        x = tf.concat([self.conv5_2(x), x], 3)
        x = tf.concat([self.conv5_3(x), x], 3)
        x = tf.concat([self.conv5_4(x), x], 3)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        ### 4th flow
        warp4 = bilinear_warp(c24, up_flow5 * 1.25)
        corr4 = CostVolumn(c1=c14, warp=warp4, search_range=4)

        x = tf.concat([corr4, c14, up_flow5, up_feat5], 3)
        x = tf.concat([self.conv4_0(x), x], 3)
        x = tf.concat([self.conv4_1(x), x], 3)
        x = tf.concat([self.conv4_2(x), x], 3)
        x = tf.concat([self.conv4_3(x), x], 3)
        x = tf.concat([self.conv4_4(x), x], 3)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        ### 3rd flow
        warp3 = bilinear_warp(c23, up_flow4 * 2.5)
        corr3 = CostVolumn(c1=c13, warp=warp3, search_range=4)

        x = tf.concat([corr3, c13, up_flow4, up_feat4], 3)
        x = tf.concat([self.conv3_0(x), x], 3)
        x = tf.concat([self.conv3_1(x), x], 3)
        x = tf.concat([self.conv3_2(x), x], 3)
        x = tf.concat([self.conv3_3(x), x], 3)
        x = tf.concat([self.conv3_4(x), x], 3)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        # 2nd flow
        warp2 = bilinear_warp(c22, up_flow3 * 5.0)
        corr2 = CostVolumn(c1=c12, warp=warp2, search_range=4)

        x = tf.concat([corr2, c12, up_flow3, up_feat3], 3)
        x = tf.concat([self.conv2_0(x), x], 3)
        x = tf.concat([self.conv2_1(x), x], 3)
        x = tf.concat([self.conv2_2(x), x], 3)
        x = tf.concat([self.conv2_3(x), x], 3)
        x = tf.concat([self.conv2_4(x), x], 3)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if is_training:
            return flow6, flow5, flow4, flow3, flow2
        else:
            self.prev_frame.assign(inputs)
            h,w = flow2.get_shape().as_list()[1:3]
            return tf.image.resize(flow2*20., [h*4, w*4])

def loss_PWCDCNet(gt_flow, pred_flows, gt_type="map"):
    def masked_reduce_mean(array, mask, axis=None):
        return tf.reduce_sum(array * mask, axis=axis) / (tf.reduce_sum(mask, axis=axis) + 1e-12)

    losses_weight = [0.32, 0.08, 0.02, 0.01, 0.005]

    flo_losses = 0.
    b, gt_h, gt_w , _ = tf.unstack(tf.shape(gt_flow))
    for flo_pred, weight in zip(pred_flows, losses_weight):
        _, pred_h, pred_w, _ = tf.unstack(tf.shape(flo_pred))

        if gt_type=="velodyne":
            # detect holes
            tmp = tf.reshape(tf.norm(gt_flow, ord=2, axis=3, keepdims=True), [b, pred_h, gt_h // pred_h, pred_w, gt_w //pred_w, 1])
            mask = tf.cast(tf.greater(tmp, 0), tf.float32)

            # resize ground-truth by taking holes into account
            tmp = tf.reshape(gt_flow, [b, pred_h, gt_h // pred_h, pred_w, gt_w //pred_w, 2])
            scaled_flow_gt = masked_reduce_mean(tmp, tf.concat([mask,mask], axis=-1), axis=[2, 4])
        else:
            scaled_flow_gt = tf.image.resize(gt_flow, tf.shape(flo_pred)[1:3], method=tf.image.ResizeMethod.BILINEAR)
        scaled_flow_gt /= 20. # tf.cast([[[[gt_h / pred_h, gt_w / pred_w]]]], dtype=tf.float32)

        mask = tf.cast(tf.greater(tf.norm(scaled_flow_gt, ord=2, axis=3), 0), tf.float32)
        l2_norm = tf.norm(flo_pred - scaled_flow_gt, ord=2, axis=3)*mask
        flo_loss = tf.reduce_mean(tf.reduce_sum(l2_norm, axis=(1, 2)))

        flo_losses += flo_loss * weight

    return flo_losses

class PWCDCNetModel(ks.models.Model):
    def __init__(self, depth_type = "map", is_training=False):

        self.is_training = is_training
        super(PWCDCNetModel, self).__init__()
        self.pwcdcnet = PWCDCNet(is_training=is_training)
        self.step_counter = tf.Variable(initial_value=tf.zeros_initializer()(shape=[], dtype='int64'), trainable=False)
        self.summaries = []
        self.depth_type = depth_type

    @tf.function
    def call(self, inputs, training=False):
        self.step_counter.assign_add(1)
        print("yes!")

        return self.pwcdcnet(inputs, self.is_training)

    @tf.function
    def train_step(self, data):
        print(data)
        with tf.name_scope("train_scope"):
            with tf.GradientTape() as tape:
                seq_len = data["depth"].get_shape().as_list()[1]
                traj_samples = [ {} for i in range(seq_len)]
                attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
                for key in attribute_list:
                    value_list = tf.unstack(data[key], axis=1)
                    for i, item in enumerate(value_list):
                        shape = item.get_shape()
                        traj_samples[i][key] = item

                inputs = tf.concat([traj_samples[-1]["RGB_im"],traj_samples[-2]["RGB_im"]], axis=-1)
                print(inputs)
                depth = traj_samples[-1]["depth"]
                rot = traj_samples[-1]["rot"]
                trans = traj_samples[-1]["trans"]
                camera = data["camera"]

                preds = self(inputs, training=True)

                gt_flow = depth2flow(tf.clip_by_value(depth, 0, 200.), rot, trans, camera)
                mask = tf.cast(tf.greater(depth, 0), tf.float32)
                loss_flow = loss_PWCDCNet(gt_flow*mask, preds, gt_type=self.depth_type)

                # Calculate the L2 norm to regularize.
                l2_losses = [0.0004 * tf.nn.l2_loss(v) for v in self.trainable_variables]
                l2_losses = tf.reduce_sum(l2_losses)

                total_losses = loss_flow + l2_losses

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_losses, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)

        gt_h, gt_w = depth.get_shape().as_list()[1:3]
        pr_h, pr_w = preds[-1].get_shape().as_list()[1:3]
        pred_flow = (tf.image.resize(preds[-1]*20., [gt_h, gt_w]))
        pred_depth = flow2depth(pred_flow, rot, trans, camera)

        with tf.name_scope("summaries"):
            max_d = 200.
            gt_d_clipped = tf.clip_by_value(traj_samples[-1]['depth'], 1., max_d)
            pr_d_clipped = tf.clip_by_value(pred_depth, 1., max_d)
            tf.summary.image("RGB_im", traj_samples[-1]['RGB_im'], step=self.step_counter)

            tf.summary.image("depth_gt", tf.math.log(gt_d_clipped) / tf.math.log(max_d), step=self.step_counter)
            tf.summary.image("depth_est", tf.math.log(pr_d_clipped) / tf.math.log(max_d), step=self.step_counter)

            im_reproj = bilinear_warp(traj_samples[-2]['RGB_im'], gt_flow)
            tf.summary.image("camera_prev_t_reproj", im_reproj, step=self.step_counter)

        with tf.name_scope("metrics"):

            max_d = 80.
            gt = tf.clip_by_value(depth, 0.000, max_d)
            est = tf.clip_by_value(pred_depth, 0.001, max_d)
            self.compiled_metrics.update_state(gt, est)
            out_dict = {m.name: m.result() for m in self.metrics}
            out_dict["loss"] = loss_flow

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return out_dict

    @tf.function
    def test_step(self, data):
        # expects one sequence element at a time (batch dim required and is free to set"
        data_format = len(data["depth"].get_shape().as_list())
        if data_format == 5:  # Sequence was recieved, eval on last frame
            seq_len = data["depth"].get_shape().as_list()[1]
            traj_samples = [{} for i in range(seq_len)]
            attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
            for key in attribute_list:
                value_list = tf.unstack(data[key], axis=1)
                for i, item in enumerate(value_list):
                    shape = item.get_shape()
                    traj_samples[i][key] = item

            for sample in traj_samples:
                pred_flow = self(sample["RGB_im"], training=False)
                eval_data = sample.copy()
            eval_data["camera"] = data["camera"]
        else:
            print(data)
            pred_flow = self(data["RGB_im"], training=False)
            eval_data = data.copy()

        with tf.name_scope("metrics"):
            depth = eval_data["depth"]
            rot = eval_data["rot"]
            trans = eval_data["trans"]
            camera = eval_data["camera"]
            gt = eval_data["depth"]
            pred_depth = flow2depth(pred_flow, rot, trans, camera)

            max_d = 80.
            gt = tf.clip_by_value(gt, 0.0, max_d)
            est = tf.clip_by_value(pred_depth, 0.001, max_d)

            if not eval_data["new_traj"]:
                self.compiled_metrics.update_state(gt, est)

        # Return a dict mapping metric names to current value.
        out_dict= {m.name: m.result() for m in self.metrics}
        return out_dict

    @tf.function
    def predict_step(self, data):
        # expects one sequence element at a time (batch dim required and is free to set"
        preds = self([[data], data["camera"]], training=False)

        with tf.name_scope("metrics"):
            est = preds

            max_d = 80.
            # est = tf.clip_by_value(est, 0.001, max_d)

            return_data={
                "image":data["RGB_im"],
                "depth": est["depth"],
                "accuracy": est["accuracy"],
                "new_traj": data["new_traj"]
            }
        return return_data