from tflearn.layers.conv import conv_2d, max_pool_2d, conv_2d_transpose
from tflearn.layers.core import input_data, dropout
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.models.dnn import DNN


# Computation Graph
def unet_model(n1, optimizer='adam', loss='categorical_crossentropy'):
    block1a = input_data(shape=[None, n1, n1, 1])
    block1a = conv_2d(block1a, 64, 3, activation='relu')
    block1a = conv_2d(block1a, 64, 3, activation='relu')

    block2a = max_pool_2d(block1a, 2, 2)

    block2a = conv_2d(block2a, 128, 3, activation='relu')
    block2a = conv_2d(block2a, 128, 3, activation='relu')

    block3a = max_pool_2d(block2a, 2, 2)

    block3a = conv_2d(block3a, 256, 3, activation='relu')
    block3a = conv_2d(block3a, 256, 3, activation='relu')

    block4a = max_pool_2d(block3a, 2, 2)

    block4a = conv_2d(block4a, 512, 3, activation='relu')
    block4a = conv_2d(block4a, 512, 3, activation='relu')
    block4a = dropout(block4a, 0.5)

    block5 = max_pool_2d(block4a, 2, 2)

    block5 = conv_2d(block5, 1024, 3, activation='relu')
    block5 = conv_2d(block5, 1024, 3, activation='relu')
    block5 = dropout(block5, 0.5)

    block4b = conv_2d_transpose(block5, 512, 3, [block5.shape[1].value * 2, block5.shape[2].value * 2, 512],
                                [1, 2, 2, 1])
    block4b = merge([block4a, block4b], 'concat', axis=3)
    block4b = conv_2d(block4b, 512, 3, activation='relu')
    block4b = conv_2d(block4b, 512, 3, activation='relu')

    block3b = conv_2d_transpose(block4b, 256, 3, [block4b.shape[1].value * 2, block4b.shape[2].value * 2, 256],
                                [1, 2, 2, 1])
    block3b = merge([block3a, block3b], 'concat', axis=3)
    block3b = conv_2d(block3b, 256, 3, activation='relu')
    block3b = conv_2d(block3b, 256, 3, activation='relu')

    block2b = conv_2d_transpose(block3b, 128, 3, [block3b.shape[1].value * 2, block3b.shape[2].value * 2, 128],
                                [1, 2, 2, 1])
    block2b = merge([block2a, block2b], 'concat', axis=3)
    block2b = conv_2d(block2b, 128, 3, activation='relu')
    block2b = conv_2d(block2b, 128, 3, activation='relu')

    block1b = conv_2d_transpose(block2b, 64, 3, [block2b.shape[1].value * 2, block2b.shape[2].value * 2, 64],
                                [1, 2, 2, 1])
    block1b = merge([block1a, block1b], 'concat', axis=3)
    block1b = conv_2d(block1b, 64, 3, activation='relu')
    block1b = conv_2d(block1b, 64, 3, activation='relu')

    Clf = conv_2d(block1b, 2, 1, 1, activation='softmax')
    regress = regression(Clf, optimizer=optimizer, loss=loss)

    return DNN(regress, tensorboard_verbose=3)
