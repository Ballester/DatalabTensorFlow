from keras.applications.vgg19 import VGG19
from keras.layers import Dense
from keras.models import Model
from keras import backend as K

def create_structure(tf, x):
    K.set_learning_phase(1)
    net = VGG19(weights='imagenet')
    base = net.get_layer('fc1').output
    pred = Dense(4096, activation='relu')(base)
    pred = Dense(1000, activation='relu')(pred)
    pred = Dense(5, activation='softmax')(pred)
    net = Model(inputs=net.input, outputs=pred)

    print(net.summary())
    return net(x)
