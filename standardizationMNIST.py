from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load minst data into X_train and X_test variables
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape images to 28X28X1
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# change the data type from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# create an image data generator, which will center the image and perform normalization
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# peform the fit on the data to do centering and standardization
datagen.fit(X_train)
# From the data generated collect images based on batch size 
for X_sample, y_sample in datagen.flow(X_train, y_train, batch_size=9):
	# prpeare a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_sample[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break