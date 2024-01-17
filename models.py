import os
import pathlib
import tensorflow as tf
def download_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME)
