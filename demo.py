import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_ubyte
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

image_strs = os.listdir('img\\')
image_strs = ['img\\' + s for s in image_strs if s[-5] == 'r'] # all color images, make cleaner later
str_splits = [s.split('-') for s in image_strs]
str_indices = {s: i for (i, s) in enumerate(image_strs)}

images = [img_as_ubyte(imread(im, as_gray=False)) for im in image_strs]

def color_img_to_scaled_temp(img):
    new_img = np.ones((601, 601), dtype=np.float) * 60 # to be converted to uint8 later
    valid = np.ones((601, 601), dtype=np.bool)

    outside = img[:, :, 3] == 0
    valid[outside] = False
    greys = np.logical_and.reduce((img[:, :, 0] == img[:, :, 1],
                                   img[:, :, 1] == img[:, :, 2],
                                   valid))
    valid[greys] = False
    light_blues = np.logical_and.reduce((img[:, :, 1] == img[:, :, 2],
                                         img[:, :, 1] > 0,
                                         valid))
    valid[light_blues] = False
    blues = np.logical_and.reduce((img[:, :, 0] == 0,
                                   img[:, :, 1] == 0,
                                   valid))
    valid[blues] = False
    greens = np.logical_and.reduce((img[:, :, 0] == 0,
                                    img[:, :, 2] == 0,
                                    valid))
    valid[greens] = False
    reds = np.logical_and.reduce((img[:, :, 1] == 0,
                                  img[:, :, 2] == 0,
                                  valid))
    valid[reds] = False
    yellows = np.logical_and.reduce((img[:, :, 0] == img[:, :, 1],
                                     valid))
    valid[yellows] = False

    if valid.any():
        plt.figure(1)
        plt.imshow(img)
        plt.figure(2)
        plt.imshow(valid)
        plt.show()
        raise(RuntimeWarning('Image has unexpected shape.'))

    scale_greys = lambda x: (x[:, :, 0] * -1 / 255 * 90) + 60
    scale_light_blues = lambda x: (x[:, :, 1] - 84) / 171 * 19 - 50
    scale_blues = lambda x: (x[:, :, 2] * -1 + 100) / 155 * 9 - 51
    scale_greens = lambda x: (x[:, :, 1] * -1 + 100) / 155 * 9 - 61
    scale_reds = lambda x: (x[:, :, 0] * -1 + 100) / 155 * 9 - 71
    scale_yellows = lambda x: (x[:, :, 1] - 79) / 176 * 9 - 90

    c_arrs = [greys, light_blues, blues, greens, reds, yellows]
    c_map_funcs = [scale_greys, scale_light_blues, scale_blues, scale_greens, scale_reds, scale_yellows]
    for i in range(6):
        temp = c_map_funcs[i](img)
        new_img[c_arrs[i]] = temp[c_arrs[i]]

    temp_to_byte = lambda x: ((x - 60) / 150 * 255 * -1).astype(np.uint8)
    return temp_to_byte(new_img)

scaled_images = [color_img_to_scaled_temp(img) for img in images]

intensities = [(float(s[-3]), float(s[-2])) for _, s in enumerate(str_splits)]

# TODO: Naive linear regression test with images completely flattened? 
flattened_imgs = [img.reshape(img.shape[0]*img.shape[1]) for img in scaled_images]
image_feats = np.array(flattened_imgs)
intensity_feats = np.array(intensities)

x_train, x_test, y_train, y_test = train_test_split(image_feats, intensity_feats, train_size=0.6)
reg = LinearRegression(fit_intercept=True, normalize=True)
reg.fit(x_train, y_train)
print(reg.score(x_train, y_train))
print(reg.score(x_test, y_test))
# It overfits to hell. Not bad.
