import keras
import numpy as np
import tensorflow as tf
from array import array
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import struct
import argparse
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)

class AutoEncoder:
    def __init__(self,batchSize,epochs,filters,layers,filter_size):
        self.batchSize = batchSize
        self.epochs = epochs
        self.layers = layers
        self.filters = filters
        self.filter_size = filter_size

    def createEncoder(self,input_img):
        temp = self.layers
        conv = input_img
        for i in range(self.layers):
            print('Shape is',conv.shape)
            conv = Conv2D(temp,kernel_size = self.filter_size , activation= 'relu',padding='same')(conv)
            conv = BatchNormalization()(conv)
            if(i == 0 or i == 1):
                conv = MaxPooling2D(pool_size=(2,2),padding='same')(conv)
            temp = temp*2

        print('Out from decoder with shape ',conv.shape)
        return conv

    def createDecoder(self,encoder):
        conv = encoder
        temp  = self.filters*(2**(self.layers))
        count = 1
        for i in range(self.layers):
            conv = Conv2D(temp, kernel_size =  self.filter_size, activation='relu',padding='same')(conv)
            conv = BatchNormalization()(conv)
            if(count == self.layers -1 or count == self.layers):
                conv = UpSampling2D((2, 2))(conv)
            count+=1
            temp = temp/2

        decoded = Conv2D(1, kernel_size =  self.filter_size, activation='sigmoid', padding='same')(conv)
        print('Out with ',decoded.shape)

        return decoded

class MyStruct:

    def __init__(self):
        self.filters = []
        self.batch = []
        self.layers = []
        self.filter_size = []
        self.epochs = []
        self.history = []

    def makeStruct(self,batch,epochs,layers,filters,filter_size,history):
        self.filters.append(filters)
        self.batch.append(batch)
        self.epochs.append(epochs)
        self.filter_size.append(filter_size)
        self.layers.append(layers)
        self.history.append(history)

parser = argparse.ArgumentParser()
parser.add_argument("-d","--data")
args = parser.parse_args()
intType = np.dtype( 'int32' ).newbyteorder( '>' )
nMetaDataBytes = 4 * intType.itemsize
data = np.fromfile(args.data, dtype = 'ubyte' )
magic, nImages, cols, rows = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
data = data[nMetaDataBytes:].astype( dtype = 'int32' ).reshape( [ nImages, cols, rows ] )
train_X,valid_X,train_ground,valid_ground = train_test_split(data,
data,
test_size=0.2,
random_state=13)
train_X = train_X/255.0
valid_X = valid_X/255.0
train_ground = train_ground/255.0
valid_ground = valid_ground/255.0
experiment = 1
Struct = MyStruct()
while True:
    batch = int(input('Give the batch size '))
    epochs = int(input('Give the number of epochs '))
    layers = int(input('Give the number of convolutional layers '))
    filters = int(input('Give the filters of each layer '))
    filter_size = tuple(int(x.strip()) for x in input('Give the filter size ').split(','))   
    x = AutoEncoder(batch,epochs,filters,layers,filter_size)   
    input_img = Input(shape = (cols,rows,1))
    
    model = Model(input_img,x.createDecoder(x.createEncoder(input_img)))
    model.compile(loss='mean_squared_error',optimizer=RMSprop())
    autoencoder_train = model.fit(train_X, train_ground, \
    batch_size=batch,epochs=epochs,verbose=1,use_multiprocessing=True,validation_data=(valid_X, valid_ground))
    Struct.makeStruct(batch,epochs,layers,filters,filter_size,autoencoder_train.history)
    while True:
        
        flag = input('Ιf you want to repeat the process with different hyperparameters please type 1\
                \nIf you want the graphs to appear on your screen please type 2\nIf you want to save the model that trained last please type 3\n')
        if(flag not in ('1','2','3')):
            print("Wrong option,please try again!")
            continue
        else:
            break

    if(flag == "1"):
        experiment+=1
        continue
    elif(flag =="2"):
        break
    elif(flag == "3"): 
        path = input('Please give the path that specifies where you want to save the model ')
        model.save_weights(path+'/autoencoder.h5')
        print("Exiting...")
        exit(0)

if(flag == "2"):
    for i in range(experiment):
        textstr = f'Epochs: {Struct.epochs[i]}, Batch size: {Struct.batch[i]}\nFilters: {Struct.filters[i]}, Filter size: {Struct.filter_size[i]}'
        fig = plt.figure()
        plt.plot(Struct.history[i]['loss'],'bo', label='(training data)')
        plt.plot(Struct.history[i]['val_loss'],'b', label='(validation data)')
        plt.ylabel('Value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        ax = plt.axes()
        ax.annotate(textstr,
                xy=(0.5, 0), xytext=(0, 10),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=10, ha='center', va='bottom')
    plt.show()

