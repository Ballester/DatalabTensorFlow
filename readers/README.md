## Readers
You should implement a class **class Dataset(object):**
This class must implement the following interface:
**def __init__(self, args={}):**
**def next_batch(self, batch_size: int):** returning a batch_size [input], [output]
**def next_test(self, batch_size: int):** returning a batch_size [input], [output]
**def get_training_size(self):** returns the training_size
**def get_test_size(self):** returns the test_size  
