import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


def generate_kernel(x, y):
    return np.ones((x,y))

def relu(x):
    if x > 0:
	    return x
    else:
        return 0

def polynomial(x, degree=3):
    return x**degree

def convolution(img, kernel):
    # Initialization of the matrix
    newMatrix = np.zeros((img.shape[0]-kernel.shape[0], img.shape[1]-kernel.shape[1]), dtype=int)
    
    for i in range(newMatrix.shape[1]):
        for j in range(newMatrix.shape[0]):
            newMatrix[j][i] = np.sum(np.multiply(img[j:kernel.shape[0]+j, i:kernel.shape[1]+i], kernel))
    return newMatrix

def display_layer(input, kernel, output):
    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.imshow(input, cmap='gray')

    if isinstance(kernel, np.ndarray):
        plt.subplot(132)
        plt.imshow(kernel, cmap='gray')

    elif isinstance(kernel, str):
        plt.suptitle(kernel)

    plt.subplot(133)
    plt.imshow(output, cmap='gray')
    plt.show()

def max_pooling(img:np.ndarray, size:tuple):
    # Initialization of the matrix
    newMatrix = np.zeros((img.shape[0]-size[0], img.shape[1]-size[1]), dtype=int)

    for i in range(newMatrix.shape[1]):
        for j in range(newMatrix.shape[0]):
            newMatrix[j][i] = np.argmax(img[j:size[0]+j, i:size[1]+i])
    return newMatrix

def activate_matrix(img, activation):
    activated = np.zeros(img.shape)
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            activated[j][i] = activation(img[j][i])
    return activated

def flatten(img):
    return img.flatten()

def test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img = x_train[5][1:]
    kernel = generate_kernel(5,5)
    conv = convolution(img, kernel)
    activated = activate_matrix(conv, polynomial)
    max_pool = max_pooling(activated, (2,2))
    flatten_matrix = flatten(img)

    display_layer(img, kernel, conv)
    display_layer(conv, "Activation", activated)
    display_layer(conv, "Max Pooling", max_pool)

test()
