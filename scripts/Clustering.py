import numpy as np
import pandas as pd
from glob import glob
import os
from skimage.io import imread
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

font = {'size'   : 14}

matplotlib.rc('font', **font)

### RGB analysis function
color_features_names = ['Gray', 'Red', 'Green', 'Blue', 'Red-Green',  'Red-Green-Sd']
def create_color_features(in_df):
    in_df['Red'] = in_df['images'].map(lambda x: np.mean(x[:,:,0]))
    in_df['Green'] = in_df['images'].map(lambda x: np.mean(x[:,:,1]))
    in_df['Blue'] = in_df['images'].map(lambda x: np.mean(x[:,:,2]))
    in_df['Gray'] = in_df['images'].map(lambda x: np.mean(x))
    in_df['Red-Green'] = in_df['images'].map(lambda x: np.mean(x[:,:,0]-x[:,:,1]))
    in_df['Red-Green-Sd'] = in_df['images'].map(lambda x: np.std(x[:,:,0]-x[:,:,1]))
    return in_df

### Clustering function
def create_color_cluster(in_df, cluster_maker = None, cluster_count = 3):
    if cluster_maker is None:
        cluster_maker = KMeans(cluster_count)
        cluster_maker.fit(in_df[['Green', 'Red-Green', 'Red-Green-Sd']])
        
    in_df['cluster-id'] = np.argmin(
        cluster_maker.transform(in_df[['Green', 'Red-Green', 'Red-Green-Sd']]),
        -1)
    in_df['cluster-id'] = in_df['cluster-id']
    # centroids = cluster_maker.cluster_centers_
    return in_df, cluster_maker

### Read in data from folders
dsb_data_dir = os.path.join('..', 'input')
stage_label = 'stage1'
all_images = glob(os.path.join(dsb_data_dir, 'stage1_*', '*', '*', '*.png'))
img_df = pd.DataFrame({'path': all_images})
img_id = lambda in_path: in_path.split('/')[-3]
img_type = lambda in_path: in_path.split('/')[-2]
img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
img_df['ImageId'] = img_df['path'].map(img_id)
img_df['ImageType'] = img_df['path'].map(img_type)
img_df['TrainingSplit'] = img_df['path'].map(img_group)
img_df['Stage'] = img_df['path'].map(img_stage)

### Take out and read in only images (no masks)
img_df = img_df.query('ImageType=="images"').drop(['ImageType'],1)
img_df['images'] = img_df['path'].map(imread)
img_df.drop(['path'],1, inplace = True)

### Do RGB analysis on all images
img_df = create_color_features(img_df)

### Split training and test sets
img_df_train = img_df.query('TrainingSplit=="train"')
img_df_test = img_df.query('TrainingSplit=="test"')

### Plot RGB distribution for training and test sets individually
# sns.set(font_scale=1.3)
sns_plot_train = sns.pairplot(img_df_train[color_features_names+['TrainingSplit']])
sns_plot_test = sns.pairplot(img_df_test[color_features_names+['TrainingSplit']])
sns_plot_train.savefig("output_train.png")
sns_plot_test.savefig("output_test.png")

### Do cluster analysis on the training set and predict the cluster for test set data
img_df_train, train_cluster_maker = create_color_cluster(img_df_train, cluster_count=4)
img_df_test, train_cluster_maker = create_color_cluster(img_df_test, train_cluster_maker, cluster_count=4)

### Plot clustered RGB data
g = clustered_train = sns.pairplot(img_df_train,
             vars = ['Green', 'Red-Green', 'Red-Green-Sd'], 
             hue = 'cluster-id',size=3)
g.fig.subplots_adjust(right=0.85)
clustered_train.savefig("clustered_train.png")
g = clustered_test = sns.pairplot(img_df_test,
             vars = ['Green', 'Red-Green', 'Red-Green-Sd'], 
             hue = 'cluster-id',size=3)
g.fig.subplots_adjust(right=0.85)
clustered_test.savefig("clustered_test.png")

### Show clustered images
n_img = 3
grouper = img_df_train.groupby(['cluster-id'])
fig, m_axs = plt.subplots(n_img, len(grouper), 
                          figsize = (16, 10))
for (c_group, clus_group), c_ims in zip(grouper, 
                                     m_axs.T):
    c_ims[0].set_title('Cluster: {}\n'.format(c_group))
    for (_, clus_row), c_im in zip(clus_group.sample(n_img, replace = True).iterrows(), c_ims):
        c_im.imshow(clus_row['images'])
        c_im.axis('off')
fig.savefig('overview_train.png')

grouper2 = img_df_test.groupby(['cluster-id'])
fig2, m_axs2 = plt.subplots(n_img, len(grouper2), 
                          figsize = (12, 10))
for (c_group, clus_group), c_ims in zip(grouper2, 
                                     m_axs2.T):
    c_ims[0].set_title('Cluster: {}\n'.format(c_group))
    for (_, clus_row), c_im in zip(clus_group.sample(n_img, replace = True).iterrows(), c_ims):
        c_im.imshow(clus_row['images'])
        c_im.axis('off')
fig2.savefig('overview_test.png')
# plt.show()

# ### Output data
# img_df_train.to_csv('clustered_stage1_train.csv')
# img_df_test.to_csv('clustered_stage1_test.csv')

