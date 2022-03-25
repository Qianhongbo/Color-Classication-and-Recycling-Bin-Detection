'''
Hand-label appropriate regions (polygonal sets of pixels) in the training images with discrete color labels.
Using the provided roipoly function.
'''

import numpy as np
import os,cv2
from roipoly import RoiPoly
from glob import glob
import pickle
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def generateColorData(folder):
    '''
    similar to generate_rgb_data
    using Roipoly (https://github.com/jdoepfert/roipoly.py)
    '''
    assert isinstance(folder, str)

    n = len(next(os.walk(folder))[2])  # number of files (60)
    X_blue = np.empty([0, 3])
    X_not_blue = np.empty([0, 3])
    for i in range(20,40):
        filename = os.listdir(folder)[i]
        # read image
        # img = plt.imread(os.path.join(folder,filename), 0)
        img = cv2.imread(os.path.join(folder, filename))
        # convert from BGR (opencv convention) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        plt.imshow(img)
        my_roi_1 = RoiPoly(color='b')
        mask = my_roi_1.get_mask(img)
        X_blue = np.concatenate([X_blue, img[mask == 1]], axis=0)

        plt.imshow(img)
        my_roi_2 = RoiPoly(color='r')
        mask = my_roi_2.get_mask(img)
        X_not_blue = np.concatenate([X_not_blue, img[mask == 1]], axis=0)

    return X_blue, X_not_blue


if __name__ == '__main__':
    folder = 'data/training'

    with open('yuv_color_data.pkl', 'ab') as f:
        X_blue, X_not_blue = generateColorData(folder)
        pickle.dump([X_blue, X_not_blue], f)

    print(X_blue.shape)
    print(X_not_blue.shape)
