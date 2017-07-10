## Post processors
A postprocessing file must have the function **def post_process(tf, args={}):** which creates changes after the model is loaded before training. The method returns a tensor for all operations that must be done to the model.
The postprocessing file **nop.py** provides no post processing and should be used on scratch training.
