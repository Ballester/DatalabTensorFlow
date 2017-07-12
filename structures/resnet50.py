from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

def create_structure(tf, x):
    K.set_learning_phase(1)
    net = ResNet50(weights='imagenet')
    base = net.output
    pred = Dense(5, activation='softmax')(base)
    net = Model(inputs=net.input, outputs=pred)
    return net(x)
