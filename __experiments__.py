import argparse
import random
import numpy

import torch
from bayes.exp_bayes import Bayes
from cnn.exp_cnn import CNN_model
from hmm.exp_hmm import HMM_Models
from utils.classes import pick_classes, get_normal_classes, get_novel_classes
from utils.data_types import DataType

# HMM experiment options
conf_dists_obs = [[10, 10], [10, 20], [20, 20], [20, 40], [50, 50], [50, 100], [100, 100]]
grid_sizes = [2, 4, 14]
fit_size = [1000, 2000, 4000, 5000, 6000]
# Confs for testing the HMMs
dists = 50
obs = 50
grid = 4
fit = 3000
hmm_accuracy = "acc"

# CNN experiment options
batch_sizes = [128]
epochs = [10]
# Confs for testing CNN
cnn_batch = 128
cnn_epoch = 20
cnn_classes = 5
cnn_accuracy = "acc"

# Bayes experiments
experiments = {
    "more_normal": {
        DataType.NORMAL: 7,
        DataType.NOVEL: 3,
        DataType.OUTLIER: 0.02,
        "cnn": "98_45"
    },
    "more_novel": {
        DataType.NORMAL: 3,
        DataType.NOVEL: 7,
        DataType.OUTLIER: 0.02,
        "cnn": "99_15"
    },
    "same_amount": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 0.02,
        "cnn": "98_95"
    },
    "less_outliers": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 0.30,
        "cnn": "98_95"
    },
    "more_outliers": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 1.30,
        "cnn": "98_95"
    },
    "same_outliers": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 1.00,
        "cnn": "98_95"
    }
}

def parse():
    parser = argparse.ArgumentParser(
        prog='Run different commands on P7_Outlier models.',
        description='Run different commands on P7_Outlier models.',
        epilog='Hi mom!'
    )

    subparsers = parser.add_subparsers(dest='model', help='Model type')

    parser_cnn = subparsers.add_parser('cnn', help='Run commands on a convolutional neural network.')
    parser_cnn.add_argument('--hyper-test', action='store_true', help='Test the hyperparameters for the CNN model.')
    parser_cnn.add_argument('--test', action='store_true', help='Test a certain configuration of CNN')

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
    global hmm_accuracy, cnn_accuracy
    args = parse()

    if args.model == 'cnn':
        if args.hyper_test:
            pick_classes(10)
            for batch in batch_sizes:
                for epoch in epochs:
                    random.seed(10)
                    torch.manual_seed(10)
                    print(f'[INFO] Initialising CNN with {batch} and {epoch}...')
                    CNN_model(get_normal_classes(), batch, epoch, 'cnns')
        else:
            random.seed(10)
            torch.manual_seed(10)
            pick_classes(cnn_classes)
            cnn_model = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', cnn_accuracy)
            cnn_model.test()
            
    elif args.model == 'hmm':
        if args.distributions:
            print("[INFO] Starting distribution tests")
            for dists_obs in conf_dists_obs:
                random.seed(10)
                torch.manual_seed(10)
                hmm_models = HMM_Models(3000, dists_obs[0], dists_obs[1], 7)
                print(f"[INFO] Finished running models with {dists_obs[0]} and {dists_obs[1]}, got hmm_accuracy of {hmm_models.hmm_accuracy}")
        elif args.grid_size:
            print("[INFO] Starting grid_size tests")
            for size in grid_sizes:
                random.seed(10)
                torch.manual_seed(10)
                hmm_models = HMM_Models(3000, 10, 10, size)
                print(f"[INFO] Finished running models with {size}, got hmm_accuracy of {hmm_models.hmm_accuracy}")
        elif args.fit_size:
            print("[INFO] Starting fit_size tests")
            for size in fit_size:
                random.seed(10)
                torch.manual_seed(10)
                hmm_models = HMM_Models(size, 10, 10, 7)
                print(f"[INFO] Finished running models with {size}, got hmm_accuracy of {hmm_models.hmm_accuracy}")
        else:
            random.seed(10)
            torch.manual_seed(10)
            print(f"[INFO] Running models with dists: {dists}, obs: {10}, grid: {grid} and fit: {fit}")
            hmm_models = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
            hmm_models.all_class_test()
            
    elif args.threshold:
        random.seed(10)
        torch.manual_seed(10)
        pick_classes(5)
        # Normal accuracy
        hmm_models = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
        hmm_accuracy = hmm_models.accuracy
        cnn_model = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', cnn_accuracy)
        cnn_accuracy = cnn_model.accuracy
        normal_hmm = hmm_models.threshold(get_normal_classes(), DataType.NORMAL, get_normal_classes())
        normal_cnn = cnn_model.threshold(DataType.NORMAL, get_normal_classes())
        print()
        random.seed(10)
        torch.manual_seed(10)
        # Novel accuracy
        hmm_models = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
        cnn_model = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', cnn_accuracy)
        novel_hmm = hmm_models.threshold(get_normal_classes(), DataType.NOVEL, get_novel_classes())
        novel_cnn = cnn_model.threshold(DataType.NOVEL, get_novel_classes())
        print()
        random.seed(10)
        torch.manual_seed(10)
        # Outlier accuracy
        hmm_models = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
        cnn_model = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', cnn_accuracy)
        outlier_hmm = hmm_models.threshold(get_normal_classes(), DataType.OUTLIER)
        outlier_cnn = cnn_model.threshold(DataType.OUTLIER)
        print()
        print(f"Normal classes: {get_normal_classes()}")
        print(f"HMM normal: {numpy.around(normal_hmm, 2)}")
        print(f"CNN normal: {numpy.around(normal_cnn, 2)}")
        print(f"HMM novel: {numpy.around(novel_hmm, 2)}")
        print(f"CNN novel: {numpy.around(novel_cnn, 2)}")
        print(f"HMM outlier: {numpy.around(outlier_hmm, 2)}")
        print(f"CNN outlier: {numpy.around(outlier_cnn, 2)}")

    elif args.model == 'bayes':
        for experiment, setup in experiments.items():
            print(experiment)
            random.seed(10)
            torch.manual_seed(10)
            pick_classes(setup[DataType.NORMAL])
            print(f"Normal classes: {get_normal_classes()}")
            print(f"Novel classes: {get_novel_classes()}")
            cnn = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', setup["cnn"])
            hmms = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
            hmms.models_for_classes(get_normal_classes)
            
            bayes = Bayes(get_normal_classes(), get_novel_classes(), setup[DataType.OUTLIER], cnn, hmms)
            bayes.run()
            bayes.save_accuracy(experiment)

if __name__ == "__main__":
    main()
