# imgRecognition
Image Recognition Program used for research "Image Recognition for Insurance" sponsored by the Society of Actuaries (SOA).

This program is a concise image recognition program to help readers quickly understand and use deep learning models for image recognition. It is intended for educational purpose.

It is coded in Python using PyTorch, which is an open source deep learning package.

The program has been tested using Python3.5 under ubuntu 16.04 LTS with Nivida GPUs

# File Structure 
**Root folder**

model-train.py This file controls the training, testing and prediction process.

libs.py        This file imports all the required packages. Users would need to install all these packages listed in the file before using the program

data.py        This file handles image data reading, transformation and data loading.

util.py        This file contains helper functions

**Subfolder**

model          This folder contains tested deep learning models (resnet, densenet, inception, and vggnet). They are directly downloaded from the PyTorch package

example        This folder contains sample files such as run log, dataset file, and output.

# Run Setting 

A few places that would need revisions to run the program

**data.py**

Line 7: DATA_DIR           The folder that contains your own image training dataset needs to be provided.

Line 10: class name        The class name (labels assigned to the images) needs to be updated based on users' requirements (what to predict)

**model-train.py**

*training*

Line 7: Net Model          Choose the model type (resnet, densenet, vgg or inception) to use. Users can also add other models in folder "model" and refer them here

Line 13: SIZE              The resoulation of the images to use for model training and prediction. Often 256 is used but other size can be used based on the memory limit and capability of the machines.

Line 145: augment          Users can choose if any augmentation shall be applied to the images before training the model. Users can choose the augment to use here.

Line 155: output folder    Users may specify the folder where all the outputs (log file, prediction results, final model, etc.) will be saved.

Line 168: batch size       Users may define how many images a batch contains. Batch processing is used to allow parallel computing.

Line 170: training dataset Users may specify the training dataset by a file that contains the location of all training images under the DATA_DIR. See /example/debug814 as an example

Line 172: augment          Users may decide whether to use the augmentation defined in function augment (Line 145) in model training

Line 176: label info       Users may provide the file that contains the label information for each image file in the training dataset

Line 223: learning rate    Users may set the learning rate schedule. For example, LR = SetRate([ (0,0.1),  (10,0.01),  (25,0.005),  (35,0.001), (40,0.0001), (45,-1)]) means a learning rate of 0.1 for the first 10 epoches, 0.01 for the next 15 epoches, 0.005 for the next 10, 0.001 for the next 5, and 0.0001 for the last 5 epoches.

Line 236: pretrained model Users may use pretrained ImageNet models as the starting point to train their own models. The location of the pretrained model can be provided there.

Line 243: skip list        Because users may try to predict different labels from pretrained models, paramters in some layers, specifically the last fully connected layers should be removed from pretrained models. Here, users need to supply the parameter names to be removed.

*predicting*

Line 319: output folder    Users provide output folder for predictions based on a calibrated model

Line 320: calibrated model Users provide the loaction of the calibrated model. If training is done together with prediction, it does not need to be changed.

Line 331: data location    The file that specifies the locations of the testing images under DATA_DIR

Line 364: augment          Augments to be applied to test images

# Data
You would need to provide the images and the labels of the images. Proprietary image data may be used to solve specific problems. Open image datasets can be used as well.

# Run Program
1. move to the root folder that contains file "model-train.py"
2. run command "python3 model-train.py" for model training and prediction.
