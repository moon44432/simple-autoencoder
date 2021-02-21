
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
import numpy as np

# hparams
batch_size = 64
original_dim = 28 ** 2
latent_dim = 2
epochs = 50
epsilon_std = 1.0


def create_encoder():
    x = Input(shape=(original_dim,), name='input')
    c1 = Conv2D(32, (5, 5), strides=2, padding='same')(x)
    c1 = LeakyReLU(c1)
    c2 = Conv2D(64, (5, 5), strides=2, padding='same')(c1)
    c2 = LeakyReLU(c2)
    f = Flatten(c2)
    z = Dense(latent_dim, name='latent')(f)
    encoder = Model(x, z, name='encoder')

    encoder.summary()
    return encoder


def create_decoder():
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    decoder_h1 = Dense(16, activation='relu', name='decoder_h1')(decoder_input)
    decoder_h2 = Dense(256, activation='relu', name='decoder_h2')(decoder_h1)
    x_decoded = Dense(original_dim, activation='sigmoid', name='flat_decoded')(decoder_h2)
    decoder = Model(decoder_input, x_decoded, name='decoder')

    decoder.summary()
    return decoder


def create_network():
    x = Input(shape=(original_dim,), name='input')

    encoder = create_encoder()
    decoder = create_decoder()

    output = decoder(encoder(x))
    ae = Model(x, output)

    return ae


if __name__ == '__main__':
    ae = create_network()
    ae.compile(optimizer='rmsprop', loss='binary_crossentropy')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    ae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size)

    encoder = create_encoder()
    decoder = create_decoder()

    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='viridis')
    plt.colorbar()
    plt.savefig('latent_space.png')
    plt.show()

    # 숫자의 2D 매니폴드 출력하기
    n = 15  # 15x15 숫자를 담은 그림
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # 잠재 공간을 가우스 분포로 가정했기 때문에 잠재 변수 z의 값을 만들기 위해
    # 일정 간격으로 떨어진 좌표를 가우스 분포의 역 CDF(ppf)를 통해 변환합니다.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig('generated.png')
    plt.show()