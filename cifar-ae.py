import multiprocessing
import tensorflow as tf
import numpy as np
# from numba import jit
from keras.datasets import cifar10
from skimage.color import rgb2gray

from sklearn.decomposition import PCA

np.random.seed(11)

width = 32
height = 32
batch_size = 10
nb_epochs = 15
code_length = 128

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)

    input_image = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    conv1 = tf.layers.conv2d(inputs=input_image,
                            filters=32,
                            kernel_size=(3,3),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=tf.nn.relu)
    conv_output = tf.contrib.layers.flatten(conv1)

    code_layer = tf.layers.dense(inputs=conv_output,
                                 units=code_length,
                                 activation=tf.nn.relu)

    code_output = tf.layers.dense(inputs=code_layer,
                                  units=(height-2)*(width-2)*3,
                                  activation=tf.nn.relu)

    deconv_input = tf.reshape(code_output, (batch_size, height-2, width-2,3))

    deconv1 = tf.layers.conv2d_transpose(inputs=deconv_input,
                                         filters=3,
                                         kernel_size=(3,3),
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activation=tf.sigmoid
                                         )

    output_images = tf.cast(tf.reshape(deconv1,
                                       (batch_size, height, width, 3))*255.0, tf.uint8)

    loss = tf.nn.l2_loss(input_image-deconv1)

    learning_rate = tf.train.exponential_decay(learning_rate=0.0005,
                                               global_step=global_step,
                                               decay_steps=int(x_train.shape[0]/(2*batch_size)),
                                               decay_rate=0.95,
                                               staircase=True)

    trainer = tf.train.RMSPropOptimizer(learning_rate)
    training_step = trainer.minimize(loss)


session = tf.InteractiveSession(graph=graph)

tf.global_variables_initializer().run()

def create_batch(t, gray=False):
    x = np.zeros((batch_size, height, width, 3 if not gray else 1), dtype = np.float32)

    for k, image in enumerate(x_train[t:t+batch_size]):
        if gray:
            x[k, :, :, :] = rgb2gray(image)
        else:
            x[k, :, :, :] = image / 255.0

    return x

for e in range(nb_epochs):
    total_loss = 0.0

    for t in range(0, x_train.shape[0], batch_size):
        feed_dict = {input_image:create_batch(t)}

        _, v_loss = session.run([training_step, loss], feed_dict=feed_dict)
        total_loss += v_loss
    print('Epoch {} - Total loss: {}'.format(e+1, total_loss))