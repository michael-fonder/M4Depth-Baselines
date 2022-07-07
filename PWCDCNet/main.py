import argparse, os
from PWCDCNet_options import PWCDCNetOptions
cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
model_opts = PWCDCNetOptions(cmdline)
cmd, test_args = cmdline.parse_known_args()
if cmd.mode == 'eval':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
import numpy as np

import dataloaders as dl
from callbacks import *
from PWCDCNet_network import *
from metrics import *

if __name__ == '__main__':
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts = PWCDCNetOptions(cmdline)
    cmd, test_args = cmdline.parse_known_args()

    # configure tensorflow gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    enable_validation = cmd.enable_validation
    try:
        # Manage GPU memory to be able to run the validation step in parallel on the same GPU
        if cmd.mode == "validation":
            print('limit memory')
            tf.config.set_logical_device_configuration(physical_devices[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=1200)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("GPUs initialization failed")
        enable_validation = False
        pass

    working_dir = os.getcwd()

    chosen_dataloader = dl.get_loader(cmd.dataset)

    seq_len = cmd.seq_len
    ckpt_dir = cmd.ckpt_dir

    if cmd.mode == 'train' or cmd.mode == 'finetune':

        print("Training on %s" % cmd.dataset)
        tf.random.set_seed(42)
        chosen_dataloader.get_dataset("train", model_opts.dataloader_settings, batch_size=cmd.batch_size)
        data = chosen_dataloader.dataset

        model = PWCDCNetModel(depth_type=chosen_dataloader.depth_type, is_training=True)

        # Initialize callbacks
        tensorboard_cbk = keras.callbacks.TensorBoard(
            log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
            write_images=False, update_freq=1200,
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
        stop_nan_cbk = ks.callbacks.TerminateOnNaN()
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir, "train"), resume_training=True)

        ''' Declare and setup optimizer '''
        lr_boundaries = [400000, 600000, 800000, 1000000]
        num_steps = 1500000 // (cmd.batch_size // 8) + 1
        lr_boundaries = [x // (cmd.batch_size // 8) for x in lr_boundaries]  # Adjust the boundaries by batch size
        lr_values = [0.0001 / (2 ** i) for i in range(len(lr_boundaries) + 1)]
        lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries, values=lr_values)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

        model.compile(optimizer=optimizer, metrics=[RootMeanSquaredLogError()])

        if enable_validation:
            val_cbk = [CustomKittiValidationCallback(cmd)]
        else:
            val_cbk = []

        if cmd.mode == 'finetune':
            nbre_epochs = model_checkpoint_cbk.resume_epoch + (20000 // chosen_dataloader.length)
        else:
            nbre_epochs = num_steps // chosen_dataloader.length

        model.fit(data, epochs=nbre_epochs + 1,
                  initial_epoch=model_checkpoint_cbk.resume_epoch,
                  callbacks=[tensorboard_cbk, model_checkpoint_cbk, stop_nan_cbk] + val_cbk)

    elif cmd.mode == 'eval' or cmd.mode == 'validation':

        if cmd.mode == "validation":
            weights_dir = os.path.join(ckpt_dir, "train")
        else:
            weights_dir = os.path.join(ckpt_dir, "best")

        print("Evaluating on %s" % cmd.dataset)
        chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        model = PWCDCNetModel(depth_type="map")

        model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
        model.compile(metrics=[AbsRelError(),
                               SqRelError(),
                               RootMeanSquaredError(),
                               RootMeanSquaredLogError(),
                               ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)])

        metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])

        # Save results accordingly
        if cmd.mode == 'validation':
            manager = BestCheckpointManager(ckpt_dir, keep_top_n=5)
            perfs = {"abs_rel": [metrics[0]], "sq_rel": [metrics[1]], "rmse": [metrics[2]], "rmsel": [metrics[3]],
                     "a1": [metrics[4]], "a2": [metrics[5]], "a3": [metrics[6]]}
            manager.update_backup(perfs)
            string = ''
            for perf in metrics:
                string += format(perf, '.4f') + "\t\t"
            with open(os.path.join(*[ckpt_dir, "validation-perfs.txt"]), 'a') as file:
                file.write(string + '\n')
        else:
            np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, fmt='%.18e', delimiter='\t',
                       newline='\n')

    elif cmd.mode == "predict":
        chosen_dataloader.get_dataset("predict", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        model = PWCDCNetModel(depth_type=chosen_dataloader.depth_type)
        model.compile()
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir, "train"), resume_training=True)
        first_sample = data.take(1)
        model.predict(first_sample, callbacks=[model_checkpoint_cbk])

        for i, sample in enumerate(data):
            estimation = model([[sample], sample["camera"]])
            print(data)
