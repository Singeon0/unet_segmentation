import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K

try:
    from preprocess_data import pre_process
    from models.basic_unet import main_unet
    from models.multiResUnet import main_multiresunet
    from models.att_unet import main_attention_unet
    from utils import predict, score, overlay_segmentation
    from models.multires_attention import main_multires_attention_unet
    drive_path = "."
except ImportError:
    print('We are on Colab')
    drive_path = "content/drive/MyDrive/Colab Notebooks/debug"


N1 = 240  # 400,320,240,160, 112
N_CHANNELS = 1  # Images are gray
INPUT_SHAPE = (N1, N1, N_CHANNELS)
N_TRAIN_PATIENTS, N_TEST_PATIENTS = 1, 1
UNET_MODEL = False
MULTIRES_MODEL = False
ATT_UNET_MODEL = False
MULTIRES_ATT_MODEL = True

# Training

file_path = f"{drive_path}/UTAH Test set"
log_path = f"{drive_path}/UTAH Test set/log"
data_path = f"{drive_path}/UTAH Test set/Utah_Training.h5"
model_path_unet = f"{drive_path}/models/CNN_models/unet.keras"
model_path_multires = f"{drive_path}/models/CNN_models/multi_res_model.keras"
model_path_att_unet = f"{drive_path}/models/CNN_models/att_unet_model.keras"
model_path_multires_att_unet = f"{drive_path}/models/CNN_models/multires att unet 112 keras.keras"

if not os.path.exists(file_path):
    pre_process(N1, data_path=drive_path, N_train_patients=N_TRAIN_PATIENTS, N_test_patients=N_TEST_PATIENTS)

# Loading data
data = h5py.File(data_path, "r")

# Split dataset into training and testing
train_images, valid_images, train_labels, valid_labels = train_test_split(np.array(data['image']), np.array(data['label']).astype(bool), test_size=0.2, random_state=42)

# hyperparameters
optimizer = 'adam'
loss = 'categorical_crossentropy'
# lr = 0.0001
batch_size = 16
epoch = 1

print(tf.config.experimental.list_physical_devices ('GPU'))

if UNET_MODEL:
    if os.path.exists(model_path_unet):
        model_name = 'UNET_MODEL'
        print(f"Model found. Loading the model {model_name}")
        model = load_model(model_path_multires)
        print("Model loaded successfully.")
        print("Starting prediction")
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)
    else:
        model_name = 'UNET_MODEL'
        print(f"Model not found")
        model = main_unet(input_shape=INPUT_SHAPE, train_images=train_images, train_labels=train_labels, valid_images=valid_images,
                              valid_labels=valid_labels, epoch=epoch, batch_size=batch_size, optimizer=optimizer, loss=loss, filename_charts=f"metrics/metrics_chart_multires.png")
        model.save(model_path_unet)
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)

if MULTIRES_MODEL:
    if os.path.exists(model_path_multires):
        model_name = 'MULTIRES_MODEL'
        print(f"Model found. Loading the model {model_name}")
        model = load_model(model_path_multires)
        print("Model loaded successfully.")
        print("Starting prediction")
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)
    else:
        model_name = 'MULTIRES_MODEL'
        print("Model not found. Training a new model...")
        model = main_multiresunet(input_shape=INPUT_SHAPE, train_images=train_images, train_labels=train_labels, valid_images=valid_images,
                              valid_labels=valid_labels, epoch=epoch, batch_size=batch_size, optimizer=optimizer, loss=loss, filename_charts=f"metrics/metrics_chart_multires.png")
        model.save(model_path_multires)
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)

if ATT_UNET_MODEL:
    if os.path.exists(model_path_att_unet):
        model_name = 'ATT_UNET_MODEL'
        print(f"Model found. Loading the model {model_name}")
        model = load_model(model_path_att_unet)
        print("Model loaded successfully.")
        print("Starting prediction")
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)
    else:
        model_name = 'ATT_UNET_MODEL'
        print("Model not found. Training a new model...")
        model = main_attention_unet(input_shape=INPUT_SHAPE, train_images=train_images, train_labels=train_labels, valid_images=valid_images,
                              valid_labels=valid_labels, epoch=epoch, batch_size=batch_size, optimizer=optimizer, loss=loss, filename_charts=f"metrics/metrics_chart_att_unet.png")
        model.save(model_path_att_unet)
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)

if MULTIRES_ATT_MODEL:
    if os.path.exists(model_path_multires_att_unet):
        model_name = 'MULTIRES_ATT_UNET_MODEL'
        print(f"Model found. Loading the model {model_name}")
        model = load_model(model_path_multires_att_unet, safe_mode=False)
        print("Model loaded successfully.")
        print("Starting prediction")
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)
    else:
        model_name = 'MULTIRES_ATT_UNET_MODEL'
        print("Model not found. Training a new model...")
        model = main_multires_attention_unet(input_shape=INPUT_SHAPE, train_images=train_images, train_labels=train_labels, valid_images=valid_images,
                              valid_labels=valid_labels, epoch=epoch, batch_size=batch_size, optimizer=optimizer, loss=loss, filename_charts=f"metrics/metrics_chart_multires_att_unet.png")
        model.save(model_path_multires_att_unet)
        predict(file_path, model, data["train.mean"], data["train.sd"], model_name=model_name)
        print("Starting evaluation")
        score(file_path, log_path, model_name=model_name)
        print("Starting overlay")
        overlay_segmentation(file_path, model_name=model_name, alpha=0.2)
