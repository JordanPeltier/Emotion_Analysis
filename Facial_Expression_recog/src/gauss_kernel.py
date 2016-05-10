import numpy as np
import cv2
from mayavi import mlab


def gauss_kernel_3d(dims, dimt, sig, tau):
    """

    :param dim: length of the 3D-Gaussian Kernel
    :param sig: standard deviation in space
    :param tau: standard deviation in time
    :return: kern : (dim, dim, dim)-ndarray which represents the gaussian-kernel
    """

    # initialisation of the kernel
    kern = np.zeros(shape=(dimt, dims, dims), dtype=float)

    # compute the kernel in the space dimension
    kernxy = cv2.getGaussianKernel(dims, sig)
    kern_space = np.outer(kernxy, kernxy)

    # compute the kernel in the time dimension
    kernt = cv2.getGaussianKernel(dimt, tau)

    # product of both to obtain the spatio-temporal kernel
    for n in range(dimt):
        kern[n] = kern_space * kernt[n]

    return kern



def plot_kern_3d(kern):
    """
    Plot a ndarray -> do not use with big arrays !! it costs a lot
    :param kern:
    :return: plot the graph
    """
    # noinspection PyUnusedLocal
    figure = mlab.figure('DensityPlot')
    mlab.points3d(kern)
    mlab.axes()
    mlab.show()
