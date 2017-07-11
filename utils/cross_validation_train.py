import os
name = "apples_cross_validation"
n_folds = 2
seed = 185
for i in range(n_folds): #number of folds
    str_format = "python main_cross_validation.py -n %d -f %d -s %d -l %s" % (n_folds, i, seed, name)
    print(str_format)
    os.system(str_format)
