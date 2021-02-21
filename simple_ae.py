import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, \
    Conv2DTranspose, LeakyReLU, Flatten, Dropout, BatchNormalization, Reshape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential, Model

# hparams
batch_size = 64
original_dim = [28, 28, 1]
latent_dim = 2
epochs = 20
epsilon_std = 1.0


def create_encoder():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                     input_shape=original_dim))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(latent_dim))

    model.summary()
    return model


def create_decoder():
    model = tf.keras.Sequential()
    model.add(Dense(7 * 7 * 64, use_bias=False, input_shape=(latent_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 64)))
    assert model.output_shape == (None, 7, 7, 64)

    model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    model.summary()
    return model


if __name__ == '__main__':
    x = Input(shape=original_dim, name='input')

    encoder = create_encoder()
    decoder = create_decoder()

    output = decoder(encoder(x))
    ae = Model(x, output)
    ae.compile(optimizer='rmsprop', loss=MeanSquaredError())

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    ae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size)
    ae.save('model.h5')

    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='viridis')
    plt.colorbar()
    plt.savefig('latent_space.png')
    plt.show()
