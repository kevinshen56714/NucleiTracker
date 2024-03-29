# NucleiTracker
An automatic nuclei detection application through convolutional neural network to detect cell neclei from biological images for [2018 Data Science Bowl on Kaggle](https://www.kaggle.com/c/data-science-bowl-2018). We created machine learning models using a convolution neural network to automate nucleus detection from images of cells. To improve model predictions, we leveraged clustering preprocessing to deal with data of various cell types, magnifications, and imaging modalities. This work gives insights into how utilizing basic data mining techniques can better train complex machine learning models.

The final report is available on [GitHub](https://github.com/kevinshen56714/NucleiTracker/blob/main/Final%20Report.pdf) or [Google Drive](https://drive.google.com/file/d/1hn8oPZVmFTLSi3PdyBT5aKUOSiKyKaxw/view?usp=sharing).

## Examples
A representative example from our predictions using CNN and U-Net:
<p align="center">
	<img src="Examples/Snapshot.png"/>
</p>

Improvement is significant when clustering pre-processing is performed prior to training the machine models, see representative results below:
<p align="center">
	<img src="Examples/Clustering1.png"/>
</p>

<p align="center">
	<img src="Examples/Clustering2.png"/>
</p>
