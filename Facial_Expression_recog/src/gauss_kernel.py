import numpy as np
import cv2
from mayavi import mlab

def gauss_kernel_3d(dim, sig, tau):
    """

    :param dim: length of the 3D-Gaussian Kernel
    :param sig: standard deviation in space
    :param tau: standard deviation in time
    :return: kern : (dim, dim, dim)-ndarray which represents the gaussian-kernel
    """

    # initialisation of the kernel
    kern = np.zeros(shape=(dim, dim, dim), dtype = float)

    # compute the kernel in the space dimension
    kernxy = cv2.getGaussianKernel(dim, sig)
    kern_space = np.outer(kernxy, kernxy)

    # compute the kernel in the time dimension
    kernt = cv2.getGaussianKernel(dim, tau)

    # product of both to obtain the spatio-temporal kernel
    for n in range(dim):
        kern[n] = kern_space*kernt[n]

    return kern

def plot_kern_3D(kern):
    figure = mlab.figure('DensityPlot')
    mlab.points3d(kern)
    mlab.axes()
    mlab.show()


# a = gauss_kernel_3d(10, np.sqrt(2), np.sqrt(2))
# b = np.sum(a, axis=(0, 1, 2))
# print a
# print b

