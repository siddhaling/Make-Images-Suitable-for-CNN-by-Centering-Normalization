# Make-Images-Suitable-for-CNN-by-Centering-Normalization
Prepare images suitable for CNN by image centering and normalization

***********************************************************************************************************************
This is a python code which reads the images from a given directory and perform centering and normalization of the images
***********************************************************************************************************************

Package Version\
python 3.6.8\
pandas 0.24.1\
Keras 2.2.4\
skimage 0.16.2

This python code is most useful to read the content of a directory\
and prepare images for suitable for CNN.\
Many times the images are not centered and not normalized to a range of values. 
The ImageDataGenerator package provided by Keras can be used to perform the centering and normalization of images.

The input to this code is the path to a directory containing images.\
The code will demonstrate the application of ImageDataGenerator on MNIST and sentiment images and display images.

### How to run this python code?
The code standardizationMNIST.py can be executed independently, which provide the output as centered and normalized images.
To run the standardizationSentimentImages.py, please provide the path to the director containing images to imFolder.\
As an example, a few images are kept in the folder 'images'.\

Few samples are kept here for complete dataset please refer\
https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

Also it is required to change to current directory using os.chdir().\
The ground truth i.e class label information is to be kept in the file groundTruthLabel.txt.

Initially the class label information is read and stored in a dataframe.\
All the paths of the images from the given directory are read and a list of paths to images is prepared into allImsPaths.\
This list is then refined to remove the paths which are not part of ground truth file.

A directory is created by storing images content and class label of the image. The prepared directory is called as imDbDict.\
The image data are loaded into variable for the further processing.\
The ImageDataGenerator object is created with featurewise_center=True, featurewise_std_normalization=True.\
The featurewise_center scales all the images to have mean of pixel values to zero.\
featurewise_std_normalization makes all the images to have mean of pixel values zero and unit variance. 
fit() function is applied on the image database. Images are collected after processed from fit() function and images in a batch size of 9 are retrieved.\
They are then dsiplayed in a grid view.

# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passioinate Researcher, Focus on Deep Learning and its applications,\
dr.siddhaling@gmail.com
