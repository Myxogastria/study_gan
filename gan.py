import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as mdl
import tensorflow.keras.layers as lyr
import time

(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

img_size = img_train.shape[1:]

img_train = img_train.reshape([-1, *img_size, 1]) / 255.0
one_hot_train = keras.utils.to_categorical(label_train, num_classes=10)

img_test = img_test.reshape([-1, *img_size, 1]) / 255.0
one_hot_test = keras.utils.to_categorical(label_test, num_classes=10)


n_class = 10
n_filter = 5
kernel_size = 5
n_feature = 2
epochs_discriminator = 10
epochs_generator = 3
# n_filter = 3
# kernel_size = 5
# n_feature = 3
# epochs = 3

# time0 = time.time()

input_image = lyr.Input(name='input_image', shape=(28, 28, 1))
filtered = lyr.Conv2D(n_filter, name='filter', kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=np.array([*img_size, 1]))(input_image)
filtered_flattened = lyr.Flatten(name='filter_flatten')(filtered)

input_label = lyr.Input(name='input_label', shape=(10, ))
feature_and_label = lyr.concatenate([filtered_flattened, input_label], name='concatenate')
similarity = lyr.Dense(1, name='similarity', activation='sigmoid')(feature_and_label)

discriminator = mdl.Model(inputs=[input_image, input_label], outputs=similarity)
discriminator.summary()

input_noise = lyr.Input(name='input_noise', shape=(n_feature, n_class))
label_noise = lyr.Lambda(lambda x: tf.keras.backend.sum(x, axis=1), name='label_nose')(input_noise)

shape_before_transconv = np.append(np.array(img_size)-kernel_size+1, n_filter)

noise_flattened = lyr.Flatten(name='noise_flatten')(input_noise)
before_transconv = lyr.Dense(shape_before_transconv.prod(), 
    activation='relu', input_shape=(None, n_feature*n_class), name='generator_dense')(noise_flattened)
before_transconv_reshaped = lyr.Reshape(shape_before_transconv, name='dense_reshape')(before_transconv)
transconv = lyr.Conv2DTranspose(1, kernel_size=(kernel_size, kernel_size), name='transconv', 
    input_shape=shape_before_transconv)(before_transconv_reshaped)

generator = mdl.Model(inputs=input_noise, outputs=transconv, name='generator')
generator.summary()


end2end = discriminator([generator(input_noise), label_noise])
gan = mdl.Model(inputs=input_noise, outputs=end2end)
gan.summary()


def make_noise(size):
    noise_matrix = np.zeros((size, n_feature, n_class))

    feature_weight = np.random.rand(size, n_feature)
    feature_weight /= feature_weight.sum(axis=1).reshape(size, 1)
    target_class = np.random.randint(0, n_class, size)

    for i in range(size):
        noise_matrix[i, :, target_class[i]] = feature_weight[i, :]
    
    return noise_matrix, keras.utils.to_categorical(target_class, num_classes=n_class)

def input_generator(size):
    while True:
        images = np.zeros((size, *img_size, 1))
        labels = np.zeros((size, n_class))

        original = np.random.randint(0, 2, size)
        original_idx = np.random.randint(0, img_train.shape[0], original.sum())
        images[original==1, :, :, :] = img_train[original_idx, :, :, :]
        labels[original==1, :] = one_hot_train[original_idx, :]

        noise, fake_label = make_noise((1-original).sum())
        images[original==0, :, :, :] = generator.predict(noise)
        labels[original==0, :] = fake_label

        yield [images, labels], original

def gan_input_generator(size):
    while True:
        yield make_noise(size), np.ones(size)


x = input_generator(4)

plt.close('all')
for i in range(10):
    print(i)
    discriminator.trainable = True
    discriminator.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    history = discriminator.fit(input_generator(2000), steps_per_epoch=10, epochs=epochs_discriminator)

    discriminator.trainable = False
    gan.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    history = gan.fit(gan_input_generator(1000), steps_per_epoch=10, epochs=epochs_generator)

# pd.DataFrame(history.history).plot()
# plt.yscale('log')
# plt.show(block=False)


a, b = next(x)
p = discriminator.predict(a)
for i in range(4):
    plt.figure()
    plt.imshow(a[0][i])
    plt.show(block=False)
    plt.title('label:{}, original:{}, discriminator:{}'.format(np.where(a[1][i]==1), b[i], p[i]))
1

