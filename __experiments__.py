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
hmm_accuracy = "49"

# CNN experiment options
batch_sizes = [128]
epochs = [10]
# Confs for testing CNN
cnn_batch = 128
cnn_epoch = 20
cnn_classes = 5
cnn_accuracy = "98_95"

# Bayes experiments
seeds = [10, 20, 30, 40, 50]
experiments = {
    "more_normal": {
        DataType.NORMAL: 7,
        DataType.NOVEL: 3,
        DataType.OUTLIER: 0.02,
        "cnn": ["98_31", "98_34", "98_8", "98_45", "98_39"]
    },
    "more_novel": {
        DataType.NORMAL: 3,
        DataType.NOVEL: 7,
        DataType.OUTLIER: 0.02,
        "cnn": ["99_25", "99_46", "99_46", "98_96", "99_69"]
    },
    "same_amount": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 0.02,
        "cnn": ["98_76", "98_55", "98_86", "98_97", "99_03"]
    },
    "less_outliers": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 0.30,
        "cnn": ["98_76", "98_55", "98_86", "98_97", "99_03"]
    },
    "more_outliers": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 1.30,
        "cnn": ["98_76", "98_55", "98_86", "98_97", "99_03"]
    },
    "same_outliers": {
        DataType.NORMAL: 5,
        DataType.NOVEL: 5,
        DataType.OUTLIER: 1.00,
        "cnn": ["98_76", "98_55", "98_86", "98_96", "99_03"]
    }
}

# Novel experiments:
seeds = [10, 20, 30, 40, 50]
novel_experiment = {
    DataType.NORMAL: 5,
    DataType.NOVEL: 5,
    DataType.OUTLIER: 0.02,
    "cnn": ["98_76", "98_55", "98_86", "98_97", "99_03"]
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
    parser_bayes.add_argument('--novel', action='store_true', help='Test the model on novelties.')

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
                    CNN_model(get_normal_classes(), batch, epoch, 10, 'cnns')
        else:
            random.seed(10)
            torch.manual_seed(10)
            pick_classes(cnn_classes)
            cnn_model = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', 10, cnn_accuracy)
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
        hmm_models = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
        hmm_models.models_for_classes(get_normal_classes())
        cnn_model = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', 10, cnn_accuracy)
        bayes = Bayes(get_normal_classes(), get_novel_classes(), 0, cnn_model, hmm_models)
        
        # Normal accuracy
        random.seed(10)
        torch.manual_seed(10)
        print("Normal")
        bayes.threshold(DataType.NORMAL, get_normal_classes())
        print()
        
        # Novel accuracy
        random.seed(10)
        torch.manual_seed(10)
        print("Novel")
        bayes.threshold(DataType.NOVEL, get_novel_classes())
        print()
        
        # Outlier accuracy
        random.seed(10)
        torch.manual_seed(10)
        print("Outlier")
        bayes.threshold(DataType.OUTLIER)

    elif args.model == 'bayes':
        if args.novel:
            for i, seed in enumerate(seeds):
                buffer_times = []
                print(seed)
                random.seed(seed)
                torch.manual_seed(10)
                pick_classes(novel_experiment[DataType.NORMAL])
                cnn = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', seed, novel_experiment["cnn"][i])
                    
                hmms = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
                hmms.models_for_classes(get_normal_classes())
                
                bayes = Bayes(get_normal_classes(), get_novel_classes(), novel_experiment[DataType.OUTLIER], cnn, hmms)
                bayes.run(True)
                bayes.save_accuracy("novels", f"{seed}")
                buffer_times.append(bayes.buffer_batches)
                print(buffer_times)
        else:    
            for i, seed in enumerate(seeds):
                buffer_times = []
                for experiment, setup in experiments.items():
                    print(experiment)
                    random.seed(seed)
                    torch.manual_seed(10)
                    pick_classes(setup[DataType.NORMAL])
                    cnn = CNN_model(get_normal_classes(), cnn_batch, cnn_epoch, 'cnns', seed, setup["cnn"][i])
                    if setup[DataType.NORMAL] == 5:
                        experiments["less_outliers"]["cnn"][i] = cnn.accuracy
                        experiments["more_outliers"]["cnn"][i] = cnn.accuracy
                        experiments["same_outliers"]["cnn"][i] = cnn.accuracy
                        
                    hmms = HMM_Models(fit, dists, obs, grid, hmm_accuracy)
                    hmms.models_for_classes(get_normal_classes())
                    
                    bayes = Bayes(get_normal_classes(), get_novel_classes(), setup[DataType.OUTLIER], cnn, hmms)
                    bayes.run(True)
                    bayes.save_accuracy(experiment, f"{seed}")
                    buffer_times.append(bayes.buffer_batches)
                print(buffer_times)
                
if __name__ == "__main__":
    main()
