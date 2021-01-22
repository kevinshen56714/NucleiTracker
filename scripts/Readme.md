1. CNN_Unet_cluster0.py, CNN_Unet_cluster1.py, CNN_Unet_cluster2.py, and CNN_Unet_raw.py would output the figure of a resized original image and a normalized image which are selected randomly.

2. CNN_Unet_cluster0.py, CNN_Unet_cluster1.py, CNN_Unet_cluster2.py would need the clustered images dataset to train and test the simple CNN and U-Net models. Since we do not want the results be overwritten, we use the separated files to save the outcomes of training models with cluster 0, 1, 2 in test set. 
In each python file, there would be 6 main parts followed by the order:
	-Resize and normalize the images to fit the train models.
	-Train the simple CNN model (You can use 'model-dsbowl2018-1.h5' as checkpoint to help train the model).
	-Show the prediction by the simple CNN model. 
	-Train the U-Net model (You can use 'model-dsbowl2018-2.h5' as checkpoint to help train the model).
	-Show the prediction by the U-Net model.
	-Save the prediction result to .csv with Run-Length Encoding format.

3. EDA.py
	- read in files and get data statistics
	- plot distribution of properties of images and masks
	- plot some scatter plot of RGB intensities

4. Clustering.py
	- read in image files and extract RGB intensities 
	- implement k-means clustering algorithm on the training set based on Green, Red-Green and the standard deviation of Red-Green
	- “predict” clusters of each test data

5. RLE_decode_superimpose_on_img.py
	- decode RLE-encoding prediction files and superimpose predictions onto original images
	
6. Precision matrices.py
	- calculate the precisions at different IoU for each prediction
