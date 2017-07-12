import os
import numpy as np
name = "apples_cross_validation"
n_folds = 5
accuracies = []
seed = 185

for i in range(n_folds): #number of folds
    str_format = "python main_cross_validation.py -n %d -f %d -s %d -l %s" % (n_folds, i, seed, name)
    print(str_format)
    os.system(str_format)
    with open(os.path.join('logs', name) + '_' + str(i) + '.txt') as fid:
        accuracies.append(float(fid.readline()))
print("Mean accuracy: " + str(np.mean(accuracies)))
print("Std dev: " + str(np.std(accuracies)))
