import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import keras

from matplotlib import pyplot
#change to current directory
os.chdir('')
#folder containing images
imFolder='images'


#ground truth i.e actual classes of the images
grndTruthPath='groundTruthLabel.txt'

#Required dimension of the input images
dim=(256,256)
input_shape = (dim[0],dim[1],3)
#number of classes
num_classes = 2

#collect all the directors and files present in the path
dirs=os.listdir(imFolder)
clNames=['Image','Sentiment']
#Create a data frame consisting of grpund truth values
clsLabels=pd.read_csv(grndTruthPath,names=clNames,delimiter='\t')
clsLabels.set_index('Image',inplace=True)

#read images from given paths and prepare images set
def createImagesSet(allImagesFoldrPath,dim,clsLabels):
    x_imageSet=np.empty((len(allImagesFoldrPath),dim[0],dim[1],3))
    imDbDict={}
    y_Set=np.empty((len(allImagesFoldrPath),1))
    for im in range(len(allImagesFoldrPath)):
        readImage=imread(allImagesFoldrPath[im])
        print(allImagesFoldrPath[im])
        imNamge=allImagesFoldrPath[im].split('\\')[-1]
        actualClass=clsLabels.loc[imNamge,'Sentiment']
        
        if (actualClass=='positive'):
            y_Set[im]=1
        else:
            y_Set[im]=0
            
        if (len(readImage.shape)>=3):
            if readImage.shape[2]>3:
                readImage=readImage[:,:,:3]            
        else:
            print(im,readImage.shape)
            readImage=gray2rgb(readImage)            
        readImage=resize(readImage,dim)
        x_imageSet[im]=readImage
        imDbDict[allImagesFoldrPath[im]]=(x_imageSet[im],y_Set[im])
    return imDbDict

#collect image names from the path list and check if its name is present in the groundTruth or not
def collectImNames(entireDb):
    imNmPresentInGrndTrth=[]
    imPathNotPresentInGrndTrth=[]
    for imPath in range(len(entireDb)):
        imNm=entireDb[imPath].split('\\')[-1]
        if imNm in clsLabels.index:
            imNmPresentInGrndTrth.append(imNm)
        else:
            imPathNotPresentInGrndTrth.append(entireDb[imPath])
    return imNmPresentInGrndTrth,imPathNotPresentInGrndTrth


#load the train and test images into two arrays of images.Convert to float type
def load_data(allImagesTrainPath,allImagesTestPath,imDbDict):
    x_trainImSet=np.empty((len(allImagesTrainPath),dim[0],dim[1],3))
    x_testImSet=np.empty((len(allImagesTestPath),dim[0],dim[1],3))
    y_trainSet=np.zeros(len(allImagesTrainPath))
    y_testSet=np.zeros(len(allImagesTestPath))
    for trnPi in range(len(allImagesTrainPath)):
        (x_trainImSet[trnPi],y_trainSet[trnPi])=imDbDict[allImagesTrainPath[trnPi]]
    
    for testPi in range(len(allImagesTestPath)):
        (x_testImSet[testPi],y_testSet[testPi])=imDbDict[allImagesTestPath[testPi]]
        
    x_trainImSet= x_trainImSet.astype('float32')
    x_testImSet= x_testImSet.astype('float32')
    y_trainSetBinary=y_trainSet
    y_testSetBinary=y_testSet
# convert class vectors to matrices as binary
    y_trainSet= keras.utils.to_categorical(y_trainSet, num_classes)
    y_testSet= keras.utils.to_categorical(y_testSet, num_classes)
    
    print('Number of train samples in Dataset: ', x_trainImSet.shape[0])
    print('Number of test samples in Dataset: ', y_testSet.shape[0])
    
    return (x_trainImSet,y_trainSet,y_trainSetBinary), (x_testImSet,y_testSet,y_testSetBinary)


#collect all the path of images
allImsPaths=[(imPath+di) for di in dirs if('txt' not in di)]
#remove images not present in ground truth table
imNmPresentInGrndTrth,imPathNotPresentInGrndTrth=collectImNames(allImsPaths)
labels=list(clsLabels.loc[imNmPresentInGrndTrth,'Sentiment'])
for rPath in imPathNotPresentInGrndTrth:
    allImsPaths.remove(rPath)

#create an image data set
imDbDict=createImagesSet(allImsPaths,dim,clsLabels)
(x_trainImSet,y_trainSet,y_trainSetBinary), (x_testImSet,y_testSet,y_testSetBinary)=load_data(allImsPaths,[],imDbDict)

# create an image data generator, which will center the image and perform normalization
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

# peform the fit on the data to do centering and standardization
datagen.fit(x_trainImSet)
# From the data generated collect images based on batch size 
for X_batch, y_batch in datagen.flow(x_trainImSet, y_trainSet, batch_size=len(x_trainImSet)):
    print('batch size is ',len(X_batch))
	# prpeare a grid of 3x3 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i])
        # show the plot
    pyplot.show()
    break