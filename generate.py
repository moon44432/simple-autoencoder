import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from simple_ae import create_encoder, create_decoder, original_dim

if __name__ == '__main__':
    x = Input(shape=original_dim, name='input')

    encoder = create_encoder()
    decoder = create_decoder()

    output = decoder(encoder(x))
    ae = Model(x, output)
    ae.load_weights('model.h5')
    ae.summary()

    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = 500 * norm.ppf(np.linspace(0.005, 0.995, n))
    grid_y = 500 * norm.ppf(np.linspace(0.005, 0.995, n))

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

    while True:
        x, y = input('Input latent vector as x y: ').split()
        xi = int(x)
        yi = int(y)

        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure = digit
        plt.figure(figsize=(1, 1))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig('generated_x_{0}_y_{1}.png'.format(xi, yi))
        plt.show()
