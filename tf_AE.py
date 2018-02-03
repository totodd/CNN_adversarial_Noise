
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from datetime import timedelta

from tensorflow.examples.tutorials.mnist import input_data
from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer


# Up-sampling 2-D Layer (deconvolutoinal Layer)
class Conv2Dtranspose(object):
    '''
      constructor's args:
          input      : input image (2D matrix)
          output_siz : output image size
          in_ch      : number of incoming image channel
          out_ch     : number of outgoing image channel
          patch_siz  : filter(patch) size
    '''

    def __init__(self, input, output_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input
        self.rows = output_siz[0]
        self.cols = output_siz[1]
        self.out_ch = out_ch
        self.activation = activation

        wshape = [patch_siz[0], patch_siz[1], out_ch, in_ch]  # note the arguments order

        w_cvt = tf.Variable(tf.truncated_normal(wshape, stddev=0.1),
                            trainable=True)
        b_cvt = tf.Variable(tf.constant(0.1, shape=[out_ch]),
                            trainable=True)
        self.batsiz = tf.shape(input)[0]
        self.w = w_cvt
        self.b = b_cvt
        self.params = [self.w, self.b]

    def output(self):
        shape4D = [self.batsiz, self.rows, self.cols, self.out_ch]
        linout = tf.nn.conv2d_transpose(self.input, self.w, output_shape=shape4D,
                                        strides=[1, 2, 2, 1], padding='SAME') + self.b
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        else:
            self.output = linout

        return self.output


# Create the model
def model(X, w_e, b_e, w_d, b_d):
    encoded = tf.sigmoid(tf.matmul(X, w_e) + b_e)
    decoded = tf.sigmoid(tf.matmul(encoded, w_d) + b_d)

    return encoded, decoded


def mk_nn_model(x_image, y_image, encoded_y_image=None):
    # Encoding phase
    conv1 = Convolution2D(x_image, (28, 28), 1, 16,
                          (3, 3), activation='relu')
    conv1_out = conv1.output()

    pool1 = MaxPooling2D(conv1_out)
    pool1_out = pool1.output()

    conv2 = Convolution2D(pool1_out, (14, 14), 16, 8,
                          (3, 3), activation='relu')
    conv2_out = conv2.output()

    pool2 = MaxPooling2D(conv2_out)
    pool2_out = pool2.output()

    conv3 = Convolution2D(pool2_out, (7, 7), 8, 1, (3, 3), activation='relu')
    conv3_out = conv3.output()

    pool3 = MaxPooling2D(conv3_out,name='encoded')
    pool3_out = pool3.output()
    encoded = pool3_out
    # at this point the representation is (1, 4, 4) i.e. 128-dimensional
    # Decoding phase
    conv_t1 = Conv2Dtranspose(pool3_out, (7, 7), 1, 8,
                              (3, 3), activation='relu')
    conv_t1_out = conv_t1.output()

    conv_t2 = Conv2Dtranspose(conv_t1_out, (14, 14), 8, 8,
                              (3, 3), activation='relu')
    conv_t2_out = conv_t2.output()

    conv_t3 = Conv2Dtranspose(conv_t2_out, (28, 28), 8, 16,
                              (3, 3), activation='relu')
    conv_t3_out = conv_t3.output()

    conv_last = Convolution2D(conv_t3_out, (28, 28), 16, 1, (3, 3),
                              activation='sigmoid')
    decoded = conv_last.output()
    print(decoded.shape)

    # decoded = tf.reshape(decoded, [-1, 784])
    cross_entropy = -1. * y_image * tf.log(decoded) - (1. - y_image) * tf.log(1. - decoded)
    loss = tf.reduce_mean(cross_entropy)
    # encod_loss = 0
    # if encoded_y_image is not None:
    encoded_cross_entropy = -1. * encoded_y_image * tf.log(encoded) - (1. - encoded_y_image) * tf.log(1. - encoded)
    encod_loss = tf.reduce_mean(encoded_cross_entropy)

    return loss, decoded, encoded, encod_loss


def plot_images(images, decoded, noise=0.0, encode=False):
    n = 10
    for i in range(n):
        # Get the i'th image and reshape the array.
        image = images[i].reshape([28,28])
        # Add the adversarial noise to the image.
        image += noise
        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)

        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image,cmap='binary', interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

        # display reconstruction
        if encode == False:
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded[i].reshape(28, 28), cmap='binary', interpolation='nearest')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded[i].reshape(4, 4), cmap='binary', interpolation='nearest')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# if __name__ == '__main__':
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

train_flag = False
# target image
target_image = mnist.train.images[0]
mnist.train.cls = np.argmax(mnist.train.labels, axis = 1)
# print(mnist.train.cls[0])
# print(target.shape)

# Variables
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 784], name='y_')

x_image = tf.reshape(x, [-1, 28, 28, 1])
y_image = tf.reshape(y_, [-1, 28, 28, 1])
encoded_y_image = tf.placeholder(tf.float32, [None, 4, 4, 1], name='encoded_y_image')

############################# noise #########################################

noise_limit = 0.5
noise_l2_weight = 0.02
ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.VARIABLES, ADVERSARY_VARIABLES]
x_noise = tf.Variable(tf.zeros([28, 28, 1]),
                      name='x_noise', trainable=False,
                      collections=collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit), name='x_noise_clip')
x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

###################################################################

loss, decoded, encoded, encod_loss = mk_nn_model(x_noisy_image, y_image, encoded_y_image)
optimizer = tf.train.AdagradOptimizer(0.1, name='adam').minimize(loss)

# optimizer_encode = tf.train.AdagradDAOptimizer(0.1).minimize(encod_loss)

############################## Optimizer for Adversarial Noise ##################

adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = encod_loss + l2_loss_noise
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2, name='adam_adversary').minimize(loss_adversary,
                                                                          var_list=adversary_variables)
tf.add_to_collection('loss_group', loss)
tf.add_to_collection('loss_group', loss_adversary)
#####################################################
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Train
sess = tf.Session()
# with tf.Session() as sess:
sess.run(init)
sess.run(tf.variables_initializer([x_noise]))


def optimize(num_iterations, adversary_target_cls=None):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = mnist.train.next_batch(128)
        if adversary_target_cls is not None:
            y_true_batch = [target_image]
            y_true_batch = np.repeat(y_true_batch, 128, axis=0)
        else:
            y_true_batch = x_batch

        feed_dict_train = {x: x_batch,
                           y_: y_true_batch}

        if adversary_target_cls is None:
            sess.run(optimizer, feed_dict=feed_dict_train)
        else:
            sess.run(optimizer_adversary, feed_dict=feed_dict_train)
            sess.run(x_noise_clip)

        # Print status every 100 iterations.
        if (i % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-set.
            if adversary_target_cls is None:
                train_loss = sess.run(loss, feed_dict=feed_dict_train)
            else:
                train_loss = sess.run(loss_adversary, feed_dict=feed_dict_train)

            msg = "Optimization Iteration: {0:>6}, Training loss: {1:>6.3f}"
            print(msg.format(i, train_loss))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))




# for i in range(1000):
#     # batch_xs, batch_ys = mnist.train.next_batch(128)
#     # feed_dic = {x: batch_xs, y_: batch_xs}
#     # sess.run(train_step,feed_dict=feed_dic)
#     # train_step.run({x: batch_xs, y_: batch_xs})
#     if i % 100 == 0:
#         train_loss = sess.run(loss,feed_dict=feed_dic)
#         # train_loss = loss.eval({x: batch_xs, y_: batch_xs})
#         print('  step, loss = %6d: %6.3f' % (i, train_loss))

# generate decoded image with test data
########################test###########################################
def get_decoded():
    test_fd = {x: mnist.test.images, y_: mnist.test.images}
    decoded_imgs = sess.run(decoded, feed_dict=test_fd)
    print('loss (test) = ', sess.run(loss_adversary, feed_dict=test_fd))
    return decoded_imgs


# def get_encoded():
#     test_fd = {x: mnist.test.images, y_: mnist.test.images}
#     encoded_imgs = sess.run(encoded, feed_dict=test_fd)
#     return encoded_imgs

def get_noise():
    # Run the TensorFlow session to retrieve the contents of
    # the x_noise variable inside the graph.
    noise = sess.run(x_noise)
    return np.squeeze(noise)


def plot_noise():
    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()
    print(noise.shape)
    # Print statistics.
    print("Noise:")
    print("- Min:", noise.min())
    print("- Max:", noise.max())
    print("- Std:", noise.std())

    # Plot the noise.
    plt.imshow(noise, interpolation='nearest', cmap='seismic',
               vmin=-1.0, vmax=1.0)
    plt.show()


def plot(fileName):
    decoded_imgs = get_decoded()
    x_test = mnist.test.images
    noise = get_noise()
    plot_images(x_test, decoded_imgs, noise)
    encoded_imgs = get_encoded()

    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(decoded_imgs[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    plt.savefig(fileName+'.png')
    plot_images(x_test, encoded_imgs,encode=True)
    np.save('encoded_target', encoded_imgs)
    plt.savefig('encoded.png')

def get_encoded():
    test_fd = {"x:0": mnist.test.images, "y_:0": mnist.test.images}
    encoded_imgs = sess.run("encoded:0", feed_dict=test_fd)
    return encoded_imgs

if train_flag:
    print('Training...')
    optimize(100)
    saver.save(sess,'./model')
    # plot('AE')
    # plot_noise()
    # plt.show()
else:
    saver = tf.train.import_meta_graph("./model.meta")
    sess = tf.Session()
    saver.restore(sess, './model')
    encoded = get_encoded()

    for i in range(1000):
        x_batch, y_true_batch = mnist.train.next_batch(128)
        y_true_batch = [encoded[0]]
        y_true_batch = np.repeat(y_true_batch, 128, axis=0)

        feed_dict_train = {"x:0": x_batch,
                           "encoded_y_image:0": y_true_batch}

        # sess.run("adam_adversary:0", feed_dict=feed_dict_train)
        # sess.run("x_noise_clip:0")

        loss = tf.get_collection('loss_group')[0]
        loss_adversary = tf.get_collection('loss_group')[1]
        optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)
        sess.run(optimizer_adversary, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if (i % 100 == 0) or (i == 1000 - 1):
            # Calculate the accuracy on the training-set.

            train_loss = sess.run(loss_adversary, feed_dict=feed_dict_train)

            msg = "Optimization Iteration: {0:>6}, Training loss: {1:>6.3f}"
            print(msg.format(i, train_loss))


    # encoded_cross_entropy = -1. * encoded_y_image * tf.log(encoded) - (1. - encoded_y_image) * tf.log(1. - encoded)
    # encod_loss = tf.reduce_mean(encoded_cross_entropy)
    # print(encoded.shape)

# optimize(2000,3)
# plot('AE+Noise')
# plot_noise()
# plt.show()
