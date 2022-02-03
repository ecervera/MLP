# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding annotation data
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, lists of corresponding dimensions, ROIs, and labels 
def readTrafficSigns(rootpath, classes, tracks):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
               list of classes to be loaded
               dictionary of tracks for each class
    Returns:   list of images, list of corresponding dimensions, ROIs, 
               labels, and filenames'''

    images = [] # images
    dims = []
    ROIs = []
    labels = [] # corresponding labels
    filenames = []
    # loop over the selected classes and tracks
    for c in classes:
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        # gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            filename = row[0]
            #if tracks[c] == int(filename[0:5]):
            if int(filename[0:5]) in tracks[c]:
                images.append(plt.imread(prefix + filename)) # the 1th column is the filename
                dims.append((int(row[1]),int(row[2])))
                ROIs.append(((int(row[3]),int(row[4])),(int(row[5]),int(row[6]))))
                labels.append(row[7]) # the 8th column is the label
                filenames.append(filename)
        gtFile.close()
    return images, dims, ROIs, labels, filenames

# from scipy.misc import imresize
from skimage.transform import resize

from numpy import histogram, interp

def histeq(im,nbr_bins=256):
    imhist, bins = histogram(im.flatten(),nbr_bins)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf 

def processImage(img, roi, dx=20, dy=20):
    crop_img = img[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
    planes = crop_img.shape[2]
    sc_img = resize(crop_img,(dx, dy, planes))
    R_img = sc_img[:,:,0]
    eq_img, cdf = histeq(R_img)
    return (eq_img-128)/256

def plotTrafficSign(img, roi, dx=20, dy=20):
    crop_img = img[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
    sc_img = resize(crop_img, (dx,dy,3))
    R_img = sc_img[:,:,0]
    eq_img, cdf = histeq(R_img)
    norm_img = 2*(eq_img - 128) / 256

    plt.subplot(231), plt.imshow(img), plt.axis('off'), plt.title('Original')
    plt.subplot(232), plt.imshow(crop_img), plt.axis('off'), plt.title('Cropped')
    plt.subplot(233), plt.imshow(sc_img), plt.axis('off'), plt.title('Scaled')
    plt.subplot(234), plt.imshow(R_img, cmap='gray'), plt.axis('off'), plt.title('Red channel')
    plt.subplot(235), plt.imshow(eq_img, cmap='gray'), plt.axis('off'), plt.title('Equalized')
    plt.subplot(236), plt.imshow(norm_img, cmap='gray'), plt.axis('off'), plt.title('Normalized');
