from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

def create_structure(tf, x):
    K.set_learning_phase(1)
    net = ResNet50(weights='imagenet')
    base = net.get_layer('flatten_1').output
    pred = Dense(512, activation='relu')(base)
    pred = Dense(5, activation='softmax')(pred)
    net = Model(inputs=net.input, outputs=pred)

    print(net.summary())
    return net(x)
