from optparse import OptionParser
parser = OptionParser()
parser.add_option("-n", "--n_folds", dest="n_folds",
                  help="cross validation folds", default=1)
parser.add_option("-f", "--current_fold", dest="fold",
                  help="fold in which the network is working", default=2)
parser.add_option("-s", "--seed", dest="seed",
                  help="seed which is used for randomization at dataset", default=-1)

(options, args) = parser.parse_args()

"""
Hyper parameters
"""
learning_rate = 0.00001
batch_size = 3
input_size = (batch_size, 227, 227, 3)
output_size = (batch_size, 5)
log_dir = 'apples'

"""
Training
"""
epochs = 1


"""
Testing
"""
cross_validation = False
n_folds = int(options.n_folds)
fold = int(options.fold)
seed = int(options.seed)
if n_folds > 1:
    cross_validation = True
