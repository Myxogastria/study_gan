import datetime
import matplotlib.pyplot as plt
import numpy as np
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
n_filter = 10
kernel_size = 5
n_feature = 3
epochs = 30
# n_filter = 3
# kernel_size = 5
# n_feature = 3
# epochs = 3

shape_before_transconv = np.append(np.array(img_size)-kernel_size+1, n_filter)

# time0 = time.time()

input_image = lyr.Input(name='input', shape=(28, 28, 1))
filtered = lyr.Conv2D(n_filter, name='filter', kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=np.array([*img_size, 1]))(input_image)
filtered_flattened = lyr.Flatten(name='filter_flatten')(filtered)
feature_unbounded_flattened = lyr.Dense(n_feature*n_class, name='feature_dense')(filtered_flattened)
feature_unbounded = lyr.Reshape((n_feature, n_class), name='feature_reshape')(feature_unbounded_flattened)

classifier = lyr.Lambda(lambda x: tf.keras.backend.softmax(tf.keras.backend.sum(x, axis=1)), name='classify')(feature_unbounded)

feature_bounded = lyr.Activation("sigmoid", name='limit')(feature_unbounded)

input_feature = lyr.Input(name='input_feature', shape=(n_feature, n_class))
feature_bounded_flattened = lyr.Flatten(name='limit_flatten')(input_feature)
before_transconv = lyr.Dense(shape_before_transconv.prod(), 
    activation='relu', input_shape=(None, n_feature*n_class), name='decoder_dense')(feature_bounded_flattened)
before_transconv_reshaped = lyr.Reshape(shape_before_transconv, name='dense_reshape')(before_transconv)
transconv = lyr.Conv2DTranspose(1, kernel_size=(kernel_size, kernel_size), name='transconv', 
    input_shape=shape_before_transconv)(before_transconv_reshaped)

generator = mdl.Model(inputs=input_feature, outputs=transconv, name='generator')
generator.summary()

encoder = mdl.Model(inputs=input_image, outputs=feature_bounded)
encoder.summary()

autogenerate = generator(feature_bounded)

autoencoder = mdl.Model(inputs=input_image, outputs=[classifier, autogenerate])
autoencoder.summary()

autoencoder.compile(optimizer=keras.optimizers.Adam(), loss=['categorical_crossentropy', 'mse'])
autoencoder.fit(img_train, [one_hot_train, img_train], epochs=epochs)

for i in range(10):
    plt.figure()
    plt.imshow(img_test[i, :, :, 0])
    plt.title(label_test[i])
    plt.show(block=False)

    plt.figure()
    [a, b] = autoencoder.predict(img_test[i:(i+1), :, :, :])
    plt.imshow(b[0, :, :, 0])
    plt.title(np.argmax(a))
    plt.show(block=False)

plt.close('all')
plt.figure()
i_plot = 1
for i_feature in range(n_feature):
    for i_class in range(n_class):
        feature_matrix = np.zeros((1, n_feature, n_class))
        feature_matrix[0, i_feature, i_class] = 1

        plt.subplot(n_feature, n_class, i_plot)
        plt.imshow(generator.predict(feature_matrix)[0, :, :, 0])
        plt.title('{}, {}'.format(i_feature, i_class))
        plt.show(block=False)
        i_plot += 1

plt.close('all')
plt.figure()
i_plot = 1
for i_feature in range(n_feature):
    for i_class in range(n_class):
        feature_matrix = np.random.rand(1, n_feature, n_class)/5
        feature_matrix[0, :, i_class] += 4/5
        feature_matrix[0, i_feature, i_class] = 1

        plt.subplot(n_feature, n_class, i_plot)
        plt.imshow(generator.predict(feature_matrix)[0, :, :, 0])
        plt.title('{}, {}'.format(i_feature, i_class))
        plt.show(block=False)
        i_plot += 1

plt.close('all')
for i_class in range(n_class):
    feature_matrix = np.zeros((1, n_feature, n_class))
    feature_matrix[0, :, i_class] = 1

    plt.figure()
    plt.imshow(generator.predict(feature_matrix)[0, :, :, 0])
    plt.title('{}, {}'.format('all features', i_class))
    plt.show(block=False)

plt.close('all')
for i_class in range(n_class):
    feature_matrix = np.random.rand(1, n_feature, n_class)/5
    feature_matrix[0, :, i_class] = 1

    plt.figure()
    plt.imshow(generator.predict(feature_matrix)[0, :, :, 0])
    plt.title('{}, {}'.format('all features', i_class))
    plt.show(block=False)

plt.close('all')
for i_class in range(n_class):
    i_img_train = img_train[label_train==i_class, :, :, :]
    i_feature_train = encoder.predict(i_img_train)

    plt.figure()
    i_plot = 1
    for plot_feature in range(n_feature):
        for plot_class in range(n_class):
            plt.subplot(n_feature, n_class, i_plot)
            plt.hist(i_feature_train[:, plot_feature, plot_class], range=(0, 1))
            plt.ylim((0, i_feature_train.shape[0]))
            plt.title('{}, {}'.format(plot_feature, plot_class))
            i_plot += 1
    plt.show(block=False)


