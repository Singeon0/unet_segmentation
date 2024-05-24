from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from keras.callbacks import EarlyStopping

try:
    from utils import create_charts
    drive_path = "."
except:
    print('we are on colab\n')
    drive_path = "content/drive/MyDrive/Colab Notebooks/debug"


def unet(input_shape):
    """
    Builds the original U-Net architecture.

    :param input_shape: tuple 1x3 of (height, width, n_channels)
    :param dropout: dropout rate
    :param batchnorm: whether to use batch normalization
    :return: the U-Net model untrained
    """
    filters = [16, 32, 64, 128, 256]
    kernelsize = 3
    upsample_size = 2

    inputs = layers.Input(input_shape)

    # Downsampling layers
    dn_1 = layers.Conv2D(filters[0], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    dn_1 = layers.Conv2D(filters[0], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(dn_1)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(dn_1)

    dn_2 = layers.Conv2D(filters[1], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(pool_1)
    dn_2 = layers.Conv2D(filters[1], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(dn_2)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(dn_2)

    dn_3 = layers.Conv2D(filters[2], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(pool_2)
    dn_3 = layers.Conv2D(filters[2], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(dn_3)
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2))(dn_3)

    dn_4 = layers.Conv2D(filters[3], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(pool_3)
    dn_4 = layers.Conv2D(filters[3], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(dn_4)
    pool_4 = layers.MaxPooling2D(pool_size=(2, 2))(dn_4)

    dn_5 = layers.Conv2D(filters[4], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(pool_4)
    dn_5 = layers.Conv2D(filters[4], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(dn_5)

    # Upsampling layers
    up_5 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_5)
    up_5 = layers.concatenate([up_5, dn_4], axis=3)
    up_conv_5 = layers.Conv2D(filters[3], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_5)
    up_conv_5 = layers.Conv2D(filters[3], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_conv_5)

    up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
    up_4 = layers.concatenate([up_4, dn_3], axis=3)
    up_conv_4 = layers.Conv2D(filters[2], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_4)
    up_conv_4 = layers.Conv2D(filters[2], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_conv_4)

    up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up_3 = layers.concatenate([up_3, dn_2], axis=3)
    up_conv_3 = layers.Conv2D(filters[1], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_3)
    up_conv_3 = layers.Conv2D(filters[1], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_conv_3)

    up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up_2 = layers.concatenate([up_2, dn_1], axis=3)
    up_conv_2 = layers.Conv2D(filters[0], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_2)
    up_conv_2 = layers.Conv2D(filters[0], (kernelsize, kernelsize), activation='relu', kernel_initializer='he_normal', padding='same')(up_conv_2)

    outputs = layers.Conv2D(2, (1, 1), activation='softmax')(up_conv_2)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    # model.summary()
    return model


def main_unet(input_shape, train_images, train_labels, valid_images, valid_labels, epoch, batch_size, optimizer, loss, filename_charts):
    """
    Main function to create, compile, and train the U-Net model.

    :param input_shape: tuple 1x3 of (height, width, n_channels)
    :param train_images: training images
    :param train_labels: training labels
    :param valid_images: validation images
    :param valid_labels: validation labels
    :param epoch: number of epochs
    :param batch_size: batch size
    :param optimizer: optimizer function
    :param loss: loss function
    :param filename_charts: path where to save the metric chart
    :return: the trained U-Net model
    """
    model = unet(input_shape)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callback for early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        patience=5,  # Number of epochs with no improvement after which training will be stopped
    )

    # Fit the model on the training set and validate on a portion of it
    history_model_unet = model.fit(
        train_images, train_labels,
        epochs=epoch,
        validation_data=(valid_images, valid_labels),  # Changed from validation_set to validation_data
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping]
    )
    create_charts(history_model_unet, filename_charts)

    return model  # Return the trained model