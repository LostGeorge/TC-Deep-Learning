import os
#import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_ubyte

image_strs = os.listdir('img\\')
image_strs = ['img\\' + s for s in image_strs if s[-5] == 'r']
str_splits = [s.split('-') for s in image_strs]
str_indices = {s: i for i, s in enumerate(image_strs)}

images = [img_as_ubyte(imread(im, as_gray=False)) for im in image_strs]

ind_to_intesnity = {i: (float(s[-3]), float(s[-2])) for i, s in enumerate(str_splits)}

# TODO: Naive linear regression test with images completely flattened?
