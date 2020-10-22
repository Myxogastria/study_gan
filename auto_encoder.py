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
dim_feature = 5

shape_before_conv = np.append(np.array(img_size)-kernel_size+1, n_filter)

time0 = time.time()

input_image = lyr.Input(shape=(28, 28, 1))
filtered = lyr.Conv2D(n_filter, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=np.array([*img_size, 1]))(input_image)
filtered_flattened = lyr.Flatten()(filtered)
feature_unbounded_flattened = lyr.Dense(dim_feature*n_class)(filtered_flattened)
feature_unbounded = lyr.Reshape((dim_feature, n_class))(feature_unbounded_flattened)

classifier = lyr.Lambda(lambda x: tf.keras.backend.softmax(tf.keras.backend.sum(x, axis=0)))(feature_unbounded)

feature_bounded = lyr.Activation("sigmoid")(feature_unbounded)
feature_bounded_flattened = lyr.Flatten()(feature_bounded)
before_transconv = lyr.Dense((np.array(img_size)-kernel_size+1).prod()*n_filter, activation='relu', input_shape=(None, dim_feature*n_class))(feature_bounded)
before_transconv_reshaped = lyr.Reshape(shape_before_conv)(before_transconv)
transconv = lyr.Conv2DTranspose(1, kernel_size=(kernel_size, kernel_size), input_shape=shape_before_conv)(before_transconv_reshaped)

model_classifier = mdl.Model(inputs=input_image, outputs=classifier)
model_classifier.predict(img_train[:10, :, :, :])

model = mdl.Model(inputs=input_image, outputs=[classifier, transconv])
model.summary()
model.compile(optimizer=keras.optimizers.Adam(), loss=['categorical_crossentropy', 'mse'])

model.fit(img_train, [one_hot_train, img_train], epochs=10)

[a, b] = model.predict(img_train[:10, :, :, :])



encoder = mdl.Sequential([
    lyr.Conv2D(n_filter, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=np.array([*img_size, 1])),
    lyr.Flatten(), 
    lyr.Dense(dim_feature*n_class, activation='relu'), 
])
encoder.summary()

activated_feature = lyr.Activation("sigmoid")(encoder)
activated_feature.summary()

generator = mdl.Sequential([
    lyr.Activation("sigmoid"), 
    lyr.Dense((np.array(img_size)-kernel_size+1).prod()*n_filter, activation='relu', input_shape=(None, dim_feature*n_class)),
    lyr.Reshape(shape_before_conv), 
    lyr.Conv2DTranspose(1, kernel_size=(kernel_size, kernel_size), input_shape=shape_before_conv)
])
generator.summary()

classifier = lyr.Dense(n_class)(encoder)


model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model.fit(img_train, one_hot_train, epochs=100, batch_size=1000)

def predict(model, img):
    return model.predict(img.reshape(1, 28, 28, 1))

def kernel_map(kernel, weight):
    output = np.zeros([weight.shape[0]+kernel.shape[0]-1, weight.shape[1]+kernel.shape[1]-1])
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            output[i:(i+kernel.shape[0]), j:(j+kernel.shape[1])] += weight[i, j]*kernel
    return output

def kernel_map_sum(kernel, weights, img_size):
    r_k, c_k, _, n_k = kernel.shape
    weights = weights.reshape(img_size[0]-r_k+1, img_size[1]-c_k+1, n_k)
    return np.stack([kernel_map(kernel[:, :, :, i].squeeze(), weights[:, :, i]) for i in range(n_k)]).sum(axis=0)

if __name__ == '__main__':
    time0 = time.time()

    model = mdl.Sequential([
        lyr.Conv2D(n_filter**2, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=np.array([*img_size, 1])),
        lyr.Flatten(), 
        lyr.Dense(n_1**2, activation='relu'), 
        lyr.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    model.fit(img_train, one_hot_train, epochs=100, batch_size=1000)

    test_loss, test_acc = model.evaluate(img_test, one_hot_test)
    print('loss: {}, accuracy:{}'.format(test_loss, test_acc))

    model.save('tf_model.h5')

    print('\nend: {}\n'.format(time.time() - time0))

