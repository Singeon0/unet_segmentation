from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping

try:
  from utils import create_charts
  drive_path = "."
except:
  print('we are on colab\n')
  drive_path = "content/drive/MyDrive/Colab Notebooks/debug"


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if (activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def MultiResBlock(U, inp, alpha=1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, activation=None,
                         padding='same')

    conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3, activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3, activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(input_shape):
    '''
    MultiResUNet

    Arguments:
        input_shape = tuple 1x3 of (height, width, n_channels)
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- MultiResUNet model
    '''

    inputs = Input(input_shape)

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32 * 2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32 * 2, 3, mresblock2)

    mresblock3 = MultiResBlock(32 * 4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32 * 4, 2, mresblock3)

    mresblock4 = MultiResBlock(32 * 8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32 * 8, 1, mresblock4)

    mresblock5 = MultiResBlock(32 * 16, pool4)

    up6 = concatenate([Conv2DTranspose(32 * 8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32 * 8, up6)

    up7 = concatenate([Conv2DTranspose(32 * 4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32 * 4, up7)

    up8 = concatenate([Conv2DTranspose(32 * 2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32 * 2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = Conv2D(2, (1, 1), activation='softmax')(mresblock9)  # Adjust 'previous_layer' accordingly

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def main_multiresunet(input_shape, train_images, train_labels, valid_images, valid_labels, epoch, batch_size, optimizer, loss, filename_charts):
    """

    :param input_shape = tuple 1x3 of (height, width, n_channels)
    :param model_path: path where to save the trained model
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

    model = MultiResUnet(input_shape)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callback for early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the validation accuracy
        patience=5,  # Number of epochs with no improvement after which training will be stopped
    )

    # Fit the model on the training set and validate on a portion of it
    history_model_unet = model.fit(
        train_images, train_labels,
        epochs=epoch,
        validation_data=(valid_images, valid_labels),
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping]  # Add the new callbacks here
    )

    create_charts(history_model_unet, filename_charts)

    return model  # the trained model

