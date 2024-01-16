import pandas as pd
import numpy as np
import os
import tflite_model_maker

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import ssd_mobilenet_v2

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Ruta al directorio que contiene las imágenes de entrenamiento
train_path = '/content/drive/MyDrive/Pest/train/images'
# Ruta al directorio que contiene las etiquetas de entrenamiento
label_train = '/content/drive/MyDrive/Pest/train/labels'

# Ruta al directorio que contiene las imágenes de validación
val_path = '/content/drive/MyDrive/Pest/val/images'
# Ruta al directorio que contiene las etiquetas de validación
label_val = '/content/drive/MyDrive/Pest/val/labels'

# Visualizando datos
names = []
nums = []
data = {'Name of class':[],'Number of samples':[]}


# Iterar sobre las clases en el directorio de entrenamiento
for class_name in os.listdir(train_path):
    class_path = os.path.join(train_path, class_name)

    # Verificar si es un directorio
    if os.path.isdir(class_path):
        # Contar el número de archivos en el directorio de la clase
        num_samples = len(os.listdir(class_path))
        names.append(class_name)
        nums.append(num_samples)

data['Name of class'] += names
data['Number of samples'] += nums
df = pd.DataFrame(data)
df
### Train the object detection model ###
modelType = 'ssd_mobilenet_v2'
modelType_uri = 'https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/frameworks/TensorFlow2/variations/ssd-mobilenet-v2/versions/1'
spec = ssd_mobilenet_v2.SsdMobileNetV2Spec(
    uri=modelType_uri,
    hparams={'max_instances_per_image': 25}
)


# Load Datasets
train_data = object_detector.DataLoader.from_pascal_voc(images_dir=train_path,annotations_dir=label_train,label_map={1: "blind-beetle",2: "corn-lepidoptera",3: "cutworm",4: "mole-cricket",5: "wireworm"})
val_data = object_detector.DataLoader.from_pascal_voc(images_dir=val_path, annotations_dir=label_val, label_map={1: "blind-beetle", 2: "corn-lepidoptera", 3: "cutworm", 4: "mole-cricket", 5: "wireworm"})

epochs = 100
batch_size = 17
max_detections = 15

# Entrenar el modelo
model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=batch_size,
    train_whole_model=False,
    epochs=epochs,
    validation_data=val_data
)

# Evaluar el modelo
eval_result = model.evaluate(val_data)

# Exportar el modelo
model.export(export_dir='.', export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

# Evaluar el modelo TFLite
model.evaluate_tflite('model.tflite', val_data)

print('Training and exporting is complete')
