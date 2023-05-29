from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
import numpy as np
import os

def rand_range(a, b):
    '''
    a utility functio to generate random float values in desired range
    '''
    return np.random.rand() * (b - a) + a

def random_affine(im):
    '''
    wrapper of Affine transformation with random scale, rotation, shear and translation parameters
    '''
    tform = AffineTransform(scale=(rand_range(0.9, 1.1), rand_range(0.9, 1.1)),
                            rotation=rand_range(-0.25, 0.25),
                            shear=rand_range(-0.15, 0.15),
                            translation=(rand_range(-im.shape[0]//15, im.shape[0]//15), 
                                         rand_range(-im.shape[1]//15, im.shape[1]//15)))
    return warp(im, tform.inverse, mode='reflect')

def random_perspective(im):
    '''
    wrapper of Projective (or perspective) transform, from 4 random points selected from 4 corners of the image within a defined region.
    '''
    region = 1/6
    A = np.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
    B = np.array([[int(rand_range(0, im.shape[1] * region)), int(rand_range(0, im.shape[0] * region))], 
                  [int(rand_range(0, im.shape[1] * region)), int(rand_range(im.shape[0] * (1-region), im.shape[0]))], 
                  [int(rand_range(im.shape[1] * (1-region), im.shape[1])), int(rand_range(im.shape[0] * (1-region), im.shape[0]))], 
                  [int(rand_range(im.shape[1] * (1-region), im.shape[1])), int(rand_range(0, im.shape[0] * region))], 
                 ])

    pt = ProjectiveTransform()
    pt.estimate(A, B)
    return warp(im, pt, output_shape=im.shape[:2])



def center_crop(image,label,output_size):
    '''
    croping the image in the center from a random margin from the borders
    '''


        # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= \
                output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape

    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))
    d1 = int(round((d - output_size[2]) / 2.))

    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

    return image, label

def RandomRotFlip(image,label):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label

def random_crop(image,label,output_size):
    '''
    croping the image in the center from a random margin from the borders
    '''
    #print('image shape_ ',np.shape(image))
        # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= \
                output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape

    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])
    d1 = np.random.randint(0, d - output_size[2])
		
    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

    return image, label

def random_intensity(im):
    '''
    rescales the intesity of the image to random interval of image intensity distribution
    '''
    return rescale_intensity(im,
                             in_range=tuple(np.percentile(im, (rand_range(0,10), rand_range(90,100)))),
                             out_range=tuple(np.percentile(im, (rand_range(0,10), rand_range(90,100)))))

def random_gamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=rand_range(0.5, 1.5))

def random_gaussian(im):
    '''
    Gaussian filter for bluring the image with random variance.
    '''
    return gaussian(im, sigma=rand_range(0, 5))
    
def random_filter(im):
    '''
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    '''
    filters = [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, random_gamma, random_gaussian, random_intensity]
    filt = random.choice(filters)
    return filt(im)

def augment(im,lab, output_size,im_size=None, center_crop_flag=0,random_affine_flag=0, random_perspective_flag=0, random_filter_flag=0, random_noise_flag=0, random_crop_flag=0, random_rotflip_flag=0,resize_flag=0):
    '''
    image augmentation by doing a sereis of transfomations on the image.
    '''
    if center_crop_flag:
        im, lab = center_crop(im,lab,output_size)
    if random_rotflip_flag:

        im, lab = RandomRotFlip(im,lab)
    if random_crop_flag:
        im, lab = random_crop(im,lab,output_size)
    if random_affine_flag:
        im = random_affine(im)
    if random_perspective_flag:
        im = random_perspective(im)
    if random_filter_flag:
        im = random_filter(im)
    if random_noise_flag:
        im = random_noise(im)
    if resize_flag:
        im = resize(im, im_size)
    return im, lab

if __name__ == "__main__":
    pass
