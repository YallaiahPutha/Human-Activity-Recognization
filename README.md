# Action Recognition
The repository builds a quick and simple code for video classification (or action recognition) using our own dataset with TensorFlow,TensorFlow Lite and Keras. A video is viewed as a 3D image or several continuous 2D images . Below are two simple neural nets models:
#LRCN
Internally LRCN temporal slice for 3D images scenerio and LSTM use several 3D kernels of size (a,b,c) and channels n, e.g., (a, b, c, n) = (3, 3, 3, 16) to convolve with video input, where videos are viewed as 3D images. dropout are also used.
#TensorFlow Lite
This lite models are used for Applications lite embedded and mobile use. After train our model with neural networks and save the model in specific path then we convert our original model to Tensorflow lite weight model
#Training & Testing
For LRCN:
For LRCN, the videos are resized as (t-dim, channels, x-dim, y-dim) = (30, 3, 224, 224) since the LRCN Model only receives RGB inputs of size (224, 224).Batch normalization, dropout are used.
dataset = 83 videos  for train our model with 3 Classes ["Falling","Loitering","Voilence"]
testdata = 10 videos for testing our model performance
In the test phase, the models are almost the same as the training phase, except that dropout has to be removed and batchnorm layer uses moving average and variance instead of mini-batch values. These are taken care by using "model.eval()".
For training and testing we have jupyter notebook for end to end conversions and model conversion to lite model
Just need to run the cells.
Humam_Action_Recognition.ipynb
#Prerequisites
Just use below command to install all requirements for this project
pip3 install -r requirements.txt
#FastAPI
FastAPI is a Python framework and set of tools that enables developers to use a REST interface to call commonly used functions to implement applications.
Here we have two APIs one for normal model and another for Tensoreflow lite model.
To the Api for Normal model with port number 12200
action-recognition/api_normal_model.py
To the Api for Normal model with port number 12201
action-recognition/tflite_api.py
