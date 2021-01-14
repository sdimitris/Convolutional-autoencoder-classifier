from matplotlib import pyplot as plt
import struct
import argparse
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.layers import Input,Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Model
from keras.models import load_model
from sklearn.metrics import classification_report
import h5py

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

        
def Encoder(input_img,filters,layers,filter_size):
    temp = layers
    conv = input_img
    for i in range(layers):
        print('Shape is',conv.shape)
        conv = Conv2D(temp,kernel_size = filter_size , activation= 'relu',padding='same')(conv)
        conv = BatchNormalization()(conv)
        if(i == 0 or i == 1):
            conv = MaxPooling2D(pool_size=(2,2),padding='same')(conv)
        temp = temp*2

    print('Out from decoder with shape ',conv.shape)
    return conv  
    
def FC(enco,num_classes,fully):
    flat = Flatten()(enco)
    den = Dense(fully,activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out



parser = argparse.ArgumentParser()
parser.add_argument("-d","--data")
parser.add_argument("-dl","--datalbls")
parser.add_argument("-t","--test")
parser.add_argument("-tl","--testlbls")
parser.add_argument("-model","--autoencoder")
args = parser.parse_args()

intType = np.dtype( 'int32' ).newbyteorder( '>' )
### read the train file ###
nMetaDataBytes = 4 * intType.itemsize
train = np.fromfile(args.data, dtype = 'ubyte' )
magic, nImages, cols, rows = np.frombuffer( train[:nMetaDataBytes].tobytes(), intType )
train = train[nMetaDataBytes:].astype( dtype = 'int32' ).reshape( [ nImages, cols, rows ] )

### read the labels ###
nMetaDataBytes = 2 * intType.itemsize
trainlbls = np.fromfile(args.datalbls, dtype = 'ubyte' )
magic, nTrainLabels = np.frombuffer( trainlbls[:nMetaDataBytes].tobytes(), intType )
trainlbls= trainlbls[nMetaDataBytes:].astype( dtype = 'int32' ) # 10000 size array with labels
#### read the test images ###
nMetaDataBytes = 4 * intType.itemsize
test = np.fromfile(args.test, dtype = 'ubyte' )
magic, nTestImages, cols, rows = np.frombuffer( test[:nMetaDataBytes].tobytes(), intType )
test = test[nMetaDataBytes:].astype( dtype = 'int32' ).reshape( [ nTestImages, cols, rows ] )

#### read test labels ###
nMetaDataBytes = 2 * intType.itemsize
testlbls = np.fromfile(args.testlbls, dtype = 'ubyte' )
magic, nTestLabels = np.frombuffer( testlbls[:nMetaDataBytes].tobytes(), intType )
testlbls = testlbls[nMetaDataBytes:].astype( dtype = 'int32' ) # 10000 size array with labels
num_classes = np.unique(trainlbls).size
train_X,valid_X,train_label,valid_label = train_test_split(train,trainlbls,test_size=0.2,random_state=13)
valid_X = valid_X/255.0
train_X = train_X/255.0
test = test/255.0


experiment = 1
Struct = MyStruct()
while True:

    ### give the hyperparameters ###
    batch = int(input('Give the batch size '))
    epochs = int(input('Give the number of epochs '))
    layers = int(input('Give the number of convolutional layers '))
    filters = int(input('Give the filters of each layer '))
    filter_size = tuple(int(x.strip()) for x in input('Give the filter size ').split(','))
    fully_connected_size = int(input('Give the size of fully connected layer '))
    input_img = Input((cols,rows,1))


    encode = Encoder(input_img,filters,layers,filter_size)

    full_model = Model(input_img,FC(encode,num_classes,fully_connected_size))
    full_model.load_weights(args.autoencoder,by_name = True) #load the autoencoder weights

        
    if(layers <= 2):             # calculate the half
        model_layers = layers*3
    else:
        model_layers = 6 + (layers-2)*2
    full_model.compile(
        optimizer=keras.optimizers.Adam(),  # Optimizer
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

        
    for layer in full_model.layers[0:model_layers]: # train only the last layer
        layer.trainable = False

    phase1 = full_model.fit(train_X, train_label, \
        batch_size=batch,epochs=epochs,verbose=1,use_multiprocessing=True,validation_data=(valid_X, valid_label))

    for layer in full_model.layers[0:model_layers]: # train only the last layer
        layer.trainable = True

    phase2 = full_model.fit(train_X, train_label, \
        batch_size=batch,epochs=epochs,verbose=1,use_multiprocessing=True,validation_data=(valid_X, valid_label))


    

    Struct.makeStruct(batch,epochs,layers,filters,filter_size,phase2.history)

    while True:
            
        flag = input('Ιf you want to repeat the process with different hyperparameters please type 1\
                    \nIf you want the graphs and the accuracy scores to appear on your screen please type 2\nIf you want to continue the classification process type 3\n')
        if(flag not in ('1','2','3')):
            print("Wrong option,please try again!")
            continue
        else:
            break
    if flag == '1':
        tf.keras.backend.clear_session()                                                                                                        
        experiment += 1
        continue
    elif flag == "2":
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
        exit(0)

    elif flag == "3":
        test_eval = full_model.evaluate(test,testlbls,verbose = 0)

        #y_pred = full_model.predict(test)
        #print(classification_report(testlbls,y_pred)) 
        print('Test loss: ',test_eval[0])
        print('Test accuracy: ',test_eval[1])
        predicted_classes = full_model.predict(test)


        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

        correct = np.where(predicted_classes==testlbls)[0]
        print ("Found correct : %d " % len(correct))
        #fig1 = plt.figure()
        for i, correct in enumerate(correct[:12]):
            plt.subplot(4,3,i+1)
            plt.imshow(test[correct].reshape(28,28), cmap='gray')
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], testlbls[correct]))
        plt.tight_layout()   
        #plt.show()

        
        fig2 = plt.figure()
        incorrect = np.where(predicted_classes!=testlbls)[0]
        print ("Found incorrect %d" % len(incorrect))
        for i, incorrect in enumerate(incorrect[:12]):
            plt.subplot(4,3,i+1)
            plt.imshow(test[incorrect].reshape(28,28), cmap='gray')
            plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], testlbls[incorrect]))
            
        plt.tight_layout()


        target_names = ["Class {}".format(i) for i in range(num_classes)]
        print(classification_report(testlbls, predicted_classes, target_names=target_names))
        plt.show()
        exit(0)
        