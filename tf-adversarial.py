import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# We also need PrettyTensor.
import prettytensor as pt


import matplotlib as mpl

# mpl.use('Agg')

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


def mk_nn_model(x, y_):
    # Encoding phase
    # x_image = tf.reshape(x, [-1, 28, 28, 1]
    x_image = x
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

    conv3 = Convolution2D(pool2_out, (7, 7), 8, 8, (3, 3), activation='relu')
    conv3_out = conv3.output()

    pool3 = MaxPooling2D(conv3_out)
    pool3_out = pool3.output()
    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    # Decoding phase
    conv_t1 = Conv2Dtranspose(pool3_out, (7, 7), 8, 8,
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

    decoded = tf.reshape(decoded, [-1, 784])
    cross_entropy = -1. * x * tf.log(decoded) - (1. - x) * tf.log(1. - decoded)
    loss = tf.reduce_mean(cross_entropy)

    return loss, decoded

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)
data.train.cls = np.argmax(data.train.labels, axis=1)

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.
        image = images[i].reshape(img_shape)

        # Add the adversarial noise to the image.
        image += noise

        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(image,
                  cmap='binary', interpolation='nearest')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

############################ placeholder ###################################

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

############################# noise #########################################

noise_limit = 0.2
noise_l2_weight = 0.02
ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.VARIABLES, ADVERSARY_VARIABLES]
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise', trainable=False,
                      collections=collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))
x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

###################### convolution net ################################
import keras
from keras.objectives import mean_squared_error
#
# model_k = keras.models.load_model('mnist_classif.h5')
# model_input_layer = model_k.layers[0].input
# model_output_layer = model_k.layers[-1].output
# y_pred = model_output_layer
# loss = tf.reduce_mean(mean_squared_error(y_true, y_pred))

loss, y_pred = mk_nn_model(x_noisy_image, y_true)

# x_pretty = pt.wrap(x_noisy_image)
# with pt.defaults_scope(activation_fn=tf.nn.relu):
#     y_pred, loss = x_pretty.\
#         conv2d(kernel=5, depth=16, name='layer_conv1').\
#         max_pool(kernel=2, stride=2).\
#         conv2d(kernel=5, depth=36, name='layer_conv2').\
#         max_pool(kernel=2, stride=2).\
#         flatten().\
#         fully_connected(size=128, name='layer_fc1').\
#         softmax_classifier(num_classes=num_classes, labels=y_true)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(loss)

############################## Optimizer for Adversarial Noise ##################

adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = loss + l2_loss_noise
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary,
                                                                          var_list=adversary_variables)

################## performance ###################
# y_pred_cls = tf.argmax(y_pred, dimension=1)
# correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

################## tensor Run #########################
session = tf.Session()
session.run(tf.global_variables_initializer())


def init_noise():
    session.run(tf.variables_initializer([x_noise]))

init_noise()
train_batch_size = 64


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    # Variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    loss, decoded = mk_nn_model(x, y_)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

    init = tf.initialize_all_variables()
    # Train
    with tf.Session() as sess:
        sess.run(init)
        print('Training...')
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(128)
            train_step.run({x: batch_xs, y_: batch_ys})
            if i % 100 == 0:
                train_loss = loss.eval({x: batch_xs, y_: batch_ys})
                print('  step, loss = %6d: %6.3f' % (i, train_loss))

        # generate decoded image with test data
        test_fd = {x: mnist.test.images, y_: mnist.test.labels}
        decoded_imgs = decoded.eval(test_fd)
        print('loss (test) = ', loss.eval(test_fd))

    x_test = mnist.test.images
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.show()
plt.savefig('mnist_ae2.png')

#
# # def optimize(num_iterations, adversary_target_cls=None):
# #     # Start-time used for printing time-usage below.
# #     start_time = time.time()
# #
# #     for i in range(num_iterations):
# #
# #         # Get a batch of training examples.
# #         # x_batch now holds a batch of images and
# #         # y_true_batch are the true labels for those images.
# #         x_batch, y_true_batch = data.train.next_batch(train_batch_size)
# #
# #         # If we are searching for the adversarial noise, then
# #         # use the adversarial target-class instead.
# #         if adversary_target_cls is not None:
# #             # The class-labels are One-Hot encoded.
# #
# #             # Set all the class-labels to zero.
# #             y_true_batch = np.zeros_like(y_true_batch)
# #
# #             # Set the element for the adversarial target-class to 1.
# #             y_true_batch[:, adversary_target_cls] = 1.0
# #
# #         # Put the batch into a dict with the proper names
# #         # for placeholder variables in the TensorFlow graph.
# #         feed_dict_train = {x: x_batch,
# #                            y_true: y_true_batch}
# #
# #         # If doing normal optimization of the neural network.
# #         if adversary_target_cls is None:
# #             # Run the optimizer using this batch of training data.
# #             # TensorFlow assigns the variables in feed_dict_train
# #             # to the placeholder variables and then runs the optimizer.
# #             session.run(optimizer, feed_dict=feed_dict_train)
# #         else:
# #             # Run the adversarial optimizer instead.
# #             # Note that we have 'faked' the class above to be
# #             # the adversarial target-class instead of the true class.
# #             session.run(optimizer_adversary, feed_dict=feed_dict_train)
# #
# #             # Clip / limit the adversarial noise. This executes
# #             # another TensorFlow operation. It cannot be executed
# #             # in the same session.run() as the optimizer, because
# #             # it may run in parallel so the execution order is not
# #             # guaranteed. We need the clip to run after the optimizer.
# #             session.run(x_noise_clip)
# #
# #         # Print status every 100 iterations.
# #         if (i % 100 == 0) or (i == num_iterations - 1):
# #             # Calculate the accuracy on the training-set.
# #             # acc = session.run(accuracy, feed_dict=feed_dict_train)
# #
# #             train_loss = loss.eval({x: x_batch, y_: y_true_batch})
# #
# #             # Message for printing.
# #             msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
# #
# #             # Print it.
# #             print(msg.format(i, acc))
# #
# #     # Ending time.
# #     end_time = time.time()
# #
# #     # Difference between start and end-times.
# #     time_dif = end_time - start_time
# #
# #     # Print the time-usage.
# #     print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
# #
#
# ########## plot ##############
# def get_noise():
#     # Run the TensorFlow session to retrieve the contents of
#     # the x_noise variable inside the graph.
#     noise = session.run(x_noise)
#
#     return np.squeeze(noise)
#
#
# def plot_noise():
#     # Get the adversarial noise from inside the TensorFlow graph.
#     noise = get_noise()
#     print(noise.shape)
#     # Print statistics.
#     print("Noise:")
#     print("- Min:", noise.min())
#     print("- Max:", noise.max())
#     print("- Std:", noise.std())
#
#     # Plot the noise.
#     plt.imshow(noise, interpolation='nearest', cmap='seismic',
#                vmin=-1.0, vmax=1.0)
#
#
# def plot_example_errors(cls_pred, correct):
#     # This function is called from print_test_accuracy() below.
#
#     # cls_pred is an array of the predicted class-number for
#     # all images in the test-set.
#
#     # correct is a boolean array whether the predicted class
#     # is equal to the true class for each image in the test-set.
#
#     # Negate the boolean array.
#     incorrect = (correct == False)
#
#     # Get the images from the test-set that have been
#     # incorrectly classified.
#     images = data.test.images[incorrect]
#
#     # Get the predicted classes for those images.
#     cls_pred = cls_pred[incorrect]
#
#     # Get the true classes for those images.
#     cls_true = data.test.cls[incorrect]
#
#     # Get the adversarial noise from inside the TensorFlow graph.
#     noise = get_noise()
#
#     # Plot the first 9 images.
#     plot_images(images=images[0:9],
#                 cls_true=cls_true[0:9],
#                 cls_pred=cls_pred[0:9],
#                 noise=noise)
# # Split the test-set into smaller batches of this size.
# test_batch_size = 256
#
#
# def print_test_accuracy(adversary_target_cls=None, show_example_errors=False,
#                         show_confusion_matrix=False):
#
#     # Number of images in the test-set.
#     num_test = len(data.test.images)
#
#     noise = get_noise()
#     # Allocate an array for the predicted classes which
#     # will be calculated in batches and filled into this array.
#     cls_pred = np.zeros(shape=num_test, dtype=np.int)
#
#     # Now calculate the predicted classes for the batches.
#     # We will just iterate through all the batches.
#     # There might be a more clever and Pythonic way of doing this.
#
#     # The starting index for the next batch is denoted i.
#     i = 0
#
#     while i < num_test:
#         # The ending index for the next batch is denoted j.
#         j = min(i + test_batch_size, num_test)
#
#         # Get the images from the test-set between index i and j.
#         images = data.test.images[i:j, :]
#
#         # Get the associated labels.
#         labels = data.test.labels[i:j, :]
#
#         # Create a feed-dict with these images and labels.
#         feed_dict = {x: images,
#                      y_true: labels}
#
#         # Calculate the predicted class using TensorFlow.
#         cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
#
#         # Set the start-index for the next batch to the
#         # end-index of the current batch.
#         i = j
#
#     # Convenience variable for the true class-numbers of the test-set.
#     if adversary_target_cls is None:
#         cls_true = data.test.cls
#     else:
#         cls_true = np.zeros_like(data.test.cls)
#         cls_true = cls_true + adversary_target_cls
#     # Create a boolean array whether each image is correctly classified.
#     correct = (cls_true == cls_pred)
#
#     # Calculate the number of correctly classified images.
#     # When summing a boolean array, False means 0 and True means 1.
#     correct_sum = correct.sum()
#
#     # Classification accuracy is the number of correctly classified
#     # images divided by the total number of images in the test-set.
#     acc = float(correct_sum) / num_test
#
#     # Print the accuracy.
#     msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
#     print(msg.format(acc, correct_sum, num_test))
#     images = data.test.images
#     plot_images(images=images[0:9],
#                 cls_true=cls_true[0:9],
#                 cls_pred=cls_pred[0:9],
#                 noise=noise)
#     # Plot some examples of mis-classifications, if desired.
#     if show_example_errors:
#         print("Example errors:")
#         plot_example_errors(cls_pred=cls_pred, correct=correct)
#
# optimize(num_iterations=1000)
# plot_noise()
# # plot_noisy_image()
# print_test_accuracy()
# plt.show()
#
# optimize(num_iterations=1000, adversary_target_cls=3)
# plot_noise()
# # plot_noisy_image()
# print_test_accuracy(adversary_target_cls=3)
# plt.show()

# optimize(num_iterations=1000)
# plot_noise()
# # plot_noisy_image()
# print_test_accuracy()
# plt.show()