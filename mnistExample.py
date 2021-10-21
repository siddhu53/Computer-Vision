import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D,Input,Dense,MaxPool2D,BatchNormalization,GlobalAvgPool2D


#tensorflow .keras.Sequential
Sequential_model = tf.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32,(3,3),activation='relu'),
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64,activation='relu'),
        Dense(10,activation='softmax')  #output layer with 10 classes

    ]
)


#functional approach : function that returns a model and it is used for building projects

def functional_model():

    my_input = Input(shape=(28,28,1))
    x = Conv2D(32,(3,3),activation='relu')(my_input)
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64,activation='relu')(x)
    x = Dense(10,activation='softmax')(x)  #output layer with 10 classes

    model = tf.keras.Model(inputs=my_input,outputs=x)

    return model


#tensorflow.keras.Model : inherit from class
class MyCustomModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32,(3,3),activation='relu')
        self.conv2 = Conv2D(64,(3,3),activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128,(3,3),activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64,activation='relu')
        self.dense2 = Dense(10,activation='softmax')  #output layer with 10 classes


    def call(self,my_input):
        
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def display_examples(examples,labels):
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0,examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]
        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img,cmap='gray')
    plt.show()
        



if __name__=='__main__':
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    print("x_train.shape=",x_train.shape)
    print("y_train.shape=",y_train.shape)
    print("x_test.shape=",x_test.shape)
    print("x_test.shape=",x_test.shape)
    
    if False:
        display_examples(x_train,y_train)

    x_train = x_train.astype('float32') / 255  # because 0 represents black and 255 represents black
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train,axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)    
    print("*************************************************")

    print("x_train.shape=",x_train.shape)
    print("y_train.shape=",y_train.shape)
    print("x_test.shape=",x_test.shape)
    print("x_test.shape=",x_test.shape)

    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)

    #model = functional_model()
    model  = MyCustomModel()  
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics='accuracy')    
    # 100 images
    # prediction: 88  correct means 88% accuracy
    # label:2 then onehot encoding :[0,0,1,0,0,0,0,0,0,0] , for normal it will 2
    
    #model training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)  #train, validation, test
    
    #Evaluation on test set
    model.evaluate(x_test,y_test,batch_size=64)
    