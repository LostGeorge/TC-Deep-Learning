import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_ubyte

image = img_as_ubyte(imread('docs\\IR_Enhancement_Curves\\irtemp_big.gif'))
image2 = img_as_ubyte(imread('img\\special_tests\wilma.png'))

plt.figure(1)
plt.imshow(image)
plt.figure(2)
plt.imshow(image2)
plt.show()
