from keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers

try:
    from utils import create_charts

    drive_path = "."
except:
    print('we are on colab\n')
    drive_path = "content/drive/MyDrive/Colab Notebooks/debug"

from models.multiResUnet import MultiResBlock, ResPath
from models.att_unet import gatingsignal, attention_block


def MultiResAttentionUNet(input_shape, alpha=1.67):
    filters = [32, 64, 128, 256, 512]
    kernelsize = 3
    upsample_size = 2

    inputs = layers.Input(input_shape)

    # Encoder path
    mresblock1 = MultiResBlock(filters[0], inputs, alpha)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(mresblock1)

    mresblock2 = MultiResBlock(filters[1], pool1, alpha)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(mresblock2)

    mresblock3 = MultiResBlock(filters[2], pool2, alpha)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(mresblock3)

    mresblock4 = MultiResBlock(filters[3], pool3, alpha)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(mresblock4)

    mresblock5 = MultiResBlock(filters[4], pool4, alpha)

    # Decoder path
    gating_5 = gatingsignal(mresblock5, filters[3])
    att_5 = attention_block(mresblock4, gating_5, filters[3])
    up5 = layers.UpSampling2D(size=(upsample_size, upsample_size))(mresblock5)
    up5 = layers.concatenate([up5, att_5], axis=3)
    mresblock6 = MultiResBlock(filters[3], up5, alpha)

    gating_4 = gatingsignal(mresblock6, filters[2])
    att_4 = attention_block(mresblock3, gating_4, filters[2])
    up4 = layers.UpSampling2D(size=(upsample_size, upsample_size))(mresblock6)
    up4 = layers.concatenate([up4, att_4], axis=3)
    mresblock7 = MultiResBlock(filters[2], up4, alpha)

    up3 = layers.UpSampling2D(size=(upsample_size, upsample_size))(mresblock7)
    respath2 = ResPath(filters[1], 2, mresblock2)
    up3 = layers.concatenate([up3, respath2], axis=3)
    mresblock8 = MultiResBlock(filters[1], up3, alpha)

    up2 = layers.UpSampling2D(size=(upsample_size, upsample_size))(mresblock8)
    respath1 = ResPath(filters[0], 1, mresblock1)
    up2 = layers.concatenate([up2, respath1], axis=3)
    mresblock9 = MultiResBlock(filters[0], up2, alpha)

    conv_final = layers.Conv2D(2, kernel_size=(1, 1))(mresblock9)
    outputs = layers.Activation('softmax')(conv_final)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    # model.summary()
    return model


def main_multires_attention_unet(input_shape, train_images, train_labels, valid_images, valid_labels, epoch, batch_size,
                                 optimizer, loss, filename_charts):
    """

    :param input_shape = tuple 1x3 of (height, width, n_channels)
    :param train_images:
    :param train_labels:
    :param valid_images:
    :param valid_labels:
    :param epoch:
    :param batch_size:
    :param optimizer:
    :param loss:
    :param filename_charts: path where to save the metric chart
    :return: the trained model
    """

    model = MultiResAttentionUNet(input_shape)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callback for early stopping
    early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the validation accuracy
        patience=7,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Show this information
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    # Fit the model on the training set and validate on a portion of it
    history_model_unet = model.fit(train_images, train_labels, epochs=epoch,
        validation_data=(valid_images, valid_labels), batch_size=batch_size, shuffle=True, callbacks=[early_stopping]
        # Add the new callbacks here
    )

    create_charts(history_model_unet, filename_charts)

    return model  # the trained model
