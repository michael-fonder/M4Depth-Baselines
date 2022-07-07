import argparse, os
from PWCDCNet_options import PWCDCNetOptions
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras

import dataloaders as dl
from callbacks import *
from PWCDCNet_network import *
from metrics import *

if __name__ == '__main__':
    tf.random.set_seed(42)

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

    kitti_params = dl.DataloaderParameters(model_opts.dataloader_settings.db_path_config,
                                           os.path.join(*[cmd.records_path, "kitti-raw-filtered", "train_data"]),
                                           4, 4, True)
    kitti_dataloader = dl.get_loader("kitti-raw")
    kitti_dataloader.get_dataset("finetune", kitti_params, batch_size=cmd.batch_size)

    midair_params = dl.DataloaderParameters(model_opts.dataloader_settings.db_path_config,
                                            os.path.join(*[cmd.records_path, "midair", "train_data"]),
                                            8, 4, True)
    midair_dataloader = dl.get_loader("midair")
    midair_dataloader.get_dataset("finetune", midair_params, batch_size=cmd.batch_size,
                                  out_size=kitti_dataloader.out_size, crop=True)

    joint_dataset_length = kitti_dataloader.length*2
    joint_dataset = tf.data.Dataset.sample_from_datasets([kitti_dataloader.dataset.repeat(), midair_dataloader.dataset.repeat()], weights=[0.5, 0.5])

    ckpt_dir = cmd.ckpt_dir

    print("Finetuning jointly on Mid-Air and Kitti")
    data = joint_dataset

    model = PWCDCNetModel(depth_type="velodyne", is_training=True)

    tensorboard_cbk = keras.callbacks.TensorBoard(
        log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
        write_images=False, update_freq=1200,
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
    stop_nan_cbk = ks.callbacks.TerminateOnNaN()
    model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])

    if enable_validation:
        val_cbk = [CustomKittiValidationCallback(cmd)]
    else:
        val_cbk = []
    print(joint_dataset_length)
    model.fit(data, epochs=model_checkpoint_cbk.resume_epoch + (20000 // joint_dataset_length) + 1,
                initial_epoch=model_checkpoint_cbk.resume_epoch,
                steps_per_epoch= joint_dataset_length,
                callbacks=[tensorboard_cbk, model_checkpoint_cbk, stop_nan_cbk] + val_cbk)
