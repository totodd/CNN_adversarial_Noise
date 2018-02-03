import matplotlib.pyplot as plt
import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer
import cifar10

cifar10.maybe_download_and_extract()

images_train, cls_train, labels_train = cifar10.load_training_data()
y = np.where(cls_train == 6)[0]
train_data = images_train[y]

images_test, cls_test, labels_test = cifar10.load_test_data()
y = np.where(cls_test == 6)[0]
test_data = images_test[y]


# def plot_images(images, decoded, noise=0.0, encode=False):
n = 10
images = train_data
for i in range(n):
    # Get the i'th image and reshape the array.
    image = images[i].reshape([32,32, 3])
    # Add the adversarial noise to the image.
    # image += noise
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


# Ensure the plot is shown correctly with multiple plots
# in a single Notebook cell.
plt.show()
