import argparse
import random

import torch
from hmm.exp_hmm import HMM_Models
from utils.classes import pick_classes, get_normal_classes, get_novel_classes
from bayes.bayes import bayes

from cnn import make_cnn

random.seed(10)
torch.manual_seed(10)
generator = torch.Generator()
generator.manual_seed(10)

# Pick normal, abnormal and novelty class labels
NO_NORMS = 5

# HMM experiment options
conf_dists_obs = [[10, 10], [10, 20], [20, 20], [20, 40], [50, 50], [50, 100], [100, 100]]
grid_sizes = [2, 4, 14]
fit_size = [1000, 2000, 4000, 5000, 6000]
#Confs for testing the HMMs
dists = 10
obs = 10
grid = 7
fit = 1000
accuracy = 32

# CNN experiment options
batch_sizes = [32, 64, 128]
epochs = [10, 20]

def parse():
    parser = argparse.ArgumentParser(
        prog='Run different commands on P7_Outlier models.',
        description='Run different commands on P7_Outlier models.',
        epilog='Hi mom!'
    )

    subparsers = parser.add_subparsers(dest='model', help='Model type')

    parser_cnn = subparsers.add_parser('cnn', help='Run commands on a convolutional neural network.')
    parser_cnn.add_argument('--hyper-test', action='store_true', help='Test the hyperparameters for the CNN model.')

    parser_hmm = subparsers.add_parser('hmm', help='Run commands on a hidden markov model.')
    parser_hmm.add_argument('--distributions', action='store_true', help='Train models with different distributions')
    parser_hmm.add_argument('--fit-size', action='store_true', help='Train models with different fit sizes')
    parser_hmm.add_argument('--grid-size', action='store_true', help='Train model with different grid sizes')
    parser_hmm.add_argument('--test', action='store_true', help='Test a certain configuration of HMMs')

    parser.add_argument('--threshold', action='store_true', help='Find the tresholds')
    
    parser_bayes = subparsers.add_parser('bayes', help='Run commands on bayes model.')
    parser_bayes.add_argument('--test', action='store_true', help='Test the model.')

    return parser.parse_args()


def main():
    args = parse()

    print('[INFO] Picking normal classes...')
    pick_classes(NO_NORMS)
    normal_classes = get_normal_classes()
    novelty_classes = get_novel_classes()

    if args.model == 'cnn':
        if args.hyper_test:
            for batch in batch_sizes:
                for epoch in epochs:
                    print(f'[INFO] Initialising CNN with {batch} and {epoch}...')
                    make_cnn.train()
        elif args.test:
            print('[INFO] Testing CNN...')
            make_cnn.test()

    elif args.model == 'hmm':
        if args.distributions:
            print("[INFO] Starting distribution tests")
            for dists_obs in conf_dists_obs:
                hmm_models = HMM_Models(3000, dists_obs[0], dists_obs[1], 7)
                print(f"[INFO] Finished running models with {dists_obs[0]} and {dists_obs[1]}, got accuracy of {hmm_models.accuracy}")
        elif args.grid_size:
            print("[INFO] Starting grid_size tests")
            for size in grid_sizes:
                hmm_models = HMM_Models(3000, 10, 10, size)
                print(f"[INFO] Finished running models with {size}, got accuracy of {hmm_models.accuracy}")
        elif args.fit_size:
            print("[INFO] Starting fit_size tests")
            for size in fit_size:
                hmm_models = HMM_Models(size, 10, 10, 7)
                print(f"[INFO] Finished running models with {size}, got accuracy of {hmm_models.accuracy}")
        else:
            print(f"[INFO] Running models with dists: {dists}, obs: {10}, grid: {grid} and fit: {fit}")
            hmm_models = HMM_Models(fit, dists, obs, grid)
            hmm_models.all_class_test()
            
    elif args.threshold:
        pass
        # grid_hmm.threshold(models, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], density=10)
        # grid_hmm.threshold(_, novelty_classes, density=10)
        # grid_hmm.threshold(models, normal_classes, density=10)
        # make_cnn.threshold([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], density=10)
        # make_cnn.threshold(normal_classes, density=10)
        # make_cnn.threshold(novelty_classes, density=10)

    elif args.model == 'bayes':
        bayes(normal_classes, novelty_classes)

if __name__ == "__main__":
    main()
