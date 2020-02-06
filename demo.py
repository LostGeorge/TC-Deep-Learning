import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte

'''
Notes for image scale (deg C)

Dark Gray ----------- (64): -31 to -41
Medium Gray -------- (112): -42 to -53
Light Gray --------- (160): -54 to -63
Black ---------------- (0): -64 to -69
White -------------- (255): -70 to -75
Cold Medium Gray --- (136): -76 to -80
Cold Dark Gray ------ (88): < -80

'''

images = os.listdir('img\\')
images = ['img\\' + i for i in images]

imgs = [img_as_ubyte(imread(i, as_gray=True)) for i in images]
plt.imshow(imgs[17])
plt.show()

