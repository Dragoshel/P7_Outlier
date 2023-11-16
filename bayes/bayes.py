# Setup values
NO_CLASSES = 10
P_CLASS = 1 / NO_CLASSES

fake_priors = [[0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8]]
 
def calculate_bayes(hmm_priors: list, cnn_priors):
    # Calculate probability of each class given the models values
    p_models = []
    for hmm_val, cnn_val in zip(hmm_priors, cnn_priors):
        class_prob = hmm_val/P_CLASS + cnn_val/P_CLASS
        p_models.append(class_prob)
    return p_models
