import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# sys.stdout.flush()
# These lists will be used to store the images.
imgs = []
masks = []
paths = []

# These lists will be used to store the image metadata that will then be used to create
# pandas dataframes.
img_data = []
mask_data = []
print('Processing images ... ')

# Loop over the training images. tqdm is used to display progress as reading
# all the images can take about 1 - 2 minutes.
for n, id_ in enumerate(train_ids):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')
    paths.append(path)
    # Get image.
    imgs.append(img)
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_area = img_width * img_height

    # Initialize counter. There is one mask for each annotated nucleus.
    nucleus_count = 1
    
    mask_ = np.zeros((img_height,img_width,1), dtype=np.uint8)
    # Loop over the mask ids, read the images and gather metadata from them.
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask = imread(path + '/masks/' + mask_file)
        mask_height = mask.shape[0]
        mask_width = mask.shape[1]
        mask_area = mask_width * img_height
        
        # Sum and divide by 255 to get the number
        # of pixels for the nucleus. Masks are grayscale.
        nucleus_area = (np.sum(mask) / 255)
        
        mask_to_img_ratio = nucleus_area / mask_area
        
        # Append to masks data list that will be used to create a pandas dataframe.
        mask_data.append([n, mask_height, mask_width, mask_area, nucleus_area, mask_to_img_ratio])
        mask = np.expand_dims(mask, axis=-1)
        mask_ = np.maximum(mask_, mask)
        
        # Increment nucleus count.
        nucleus_count += 1

    masks.append(mask_)

    # Build image info list that will be used to create dataframe. This is done after the masks loop
    # because we want to store the number of nuclei per image in the img_data list.
    img_data.append([img_height, img_width, img_area, nucleus_count])

# Create dataframe for images
df_img = pd.DataFrame(img_data, columns=['height', 'width', 'area', 'nuclei'])
# print df_img.describe()

df_mask = pd.DataFrame(mask_data, columns=['img_index', 'height', 'width', 'area', 'nucleus_area', 'mask_to_img_ratio'])
# print df_mask.describe()
sns.set(font_scale=1.2)
fig, ax = plt.subplots(1, 3, figsize=(15,5))
width_plt = sns.distplot(df_img['width'].values, ax=ax[0])
width_plt.set(xlabel='width (px)')
width_plt.set(ylim=(0, 0.012))
height_plt = sns.distplot(df_img['height'].values, ax=ax[1])
height_plt.set(xlabel='height (px)')
height_plt.set(ylim=(0, 0.016))
nu_plt = sns.distplot(df_img['nuclei'].values, ax=ax[2])
nu_plt.set(xlabel='nuclei count')
fig.show()
plt.tight_layout()

# sns.distplot(df_img['nuclei'].values)
# plt.xlabel("nuclei count")
# plt.show()
# plt.figure(figsize=(18, 18))
much_nuclei = df_img['nuclei'].argmax()
print much_nuclei, paths[much_nuclei]
# plt.grid(None)
# plt.imshow(imgs[much_nuclei])

fig, ax = plt.subplots(1, 2)
ax[0].grid(None)
ax[0].imshow(imgs[much_nuclei])
ax[1].grid(None)
ax[1].imshow(np.squeeze(masks[much_nuclei]))
plt.tight_layout()

# plt.figure(figsize=(18, 18))
not_much_nuclei = df_img['nuclei'].argmin()
print df_img['nuclei'].min(), paths[not_much_nuclei]
# plt.grid(None)
# plt.imshow(imgs[not_much_nuclei])

fig, ax = plt.subplots(1, 2)
ax[0].grid(None)
ax[0].imshow(imgs[not_much_nuclei])
ax[1].grid(None)
ax[1].imshow(np.squeeze(masks[not_much_nuclei]))
plt.tight_layout()


smallest_mask_index = df_mask['mask_to_img_ratio'].argmin()
smallest_mask_img_index = df_mask.iloc[[smallest_mask_index], [0]].values[0][0]

fig, ax = plt.subplots(1, 2)
ax[0].grid(None)
# ax[0].imshow(masks[smallest_mask_index])
ax[0].imshow(np.squeeze(masks[smallest_mask_img_index]))
ax[1].grid(None)
ax[1].imshow(imgs[smallest_mask_img_index])
plt.tight_layout()

biggest_mask_index = df_mask['mask_to_img_ratio'].argmax()
biggest_mask_img_index = df_mask.iloc[[biggest_mask_index], [0]].values[0][0]

fig, ax = plt.subplots(1, 2)
ax[0].grid(None)
ax[1].grid(None)
# ax[0].imshow(masks[biggest_mask_index])
ax[0].imshow(np.squeeze(masks[biggest_mask_img_index]))
ax[1].imshow(imgs[biggest_mask_img_index])
plt.tight_layout()

plt.show()