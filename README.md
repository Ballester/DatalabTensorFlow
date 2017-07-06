# DatalabTensorFlow
Datalab UFPel TensorFlow Framework

## Tutorial
Here you can download a dataset and train your first network. This steps make you download the apples diseases dataset and a pre-trained Alexnet model.

```shell
python utils/download_apples_dataset.py
python utils/download_alexnet_weights.py
python main.py
```

The framework is organized as follows:

## Interface
All interfaces that should be implemented by a file are explained in their folder in a README.md file.

## main.py
Responsible for training and packing together the information from other files.
By importing the files from other folders, you change how the training happens. For example, if you want to change the loaded structure, you should change the line from **structures.alexnet import create_structure** to **structures.your_structure import create_structure**.

## config folder
Contains configuration files, such as learning_rate and other parameters. New config files must implement an interface.

## readers folder
Contains reader files, the ones that do in fact read the dataset. New reader files must implement an interface.

## structures folder
Contains structure files, the ones that define the networks' architecture. Must implement an interface and return the last tensor executed (tensorflow enables us to abstract the rest of the architecture as it is implicitly processed by the tensor graph).

## postprocessing folder
The trickiest folder. This one allows you to make changes in a structure after it is loaded. For example, you can create a structure in the structure folder and then add loaded weights by calling the postprocessing. New postprocessing files must implement an interface.

## utils folder
Contains utility functions

## Todo list
+ ???
