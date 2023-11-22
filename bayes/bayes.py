
NO_CLASSES = 10
P_CLASS = 1 / NO_CLASSES

hmm = [0.6, 0.15, 0.25]
cnn = [0.68, 0.22,0.1]
bayes = [0.5, 0.5]
def calculate_bayes(hmm_priors: list, cnn_priors):
    # Calculate probability of each class given the models values
    p_models = []
    for hmm_val, cnn_val, bayes_val in zip(hmm_priors, cnn_priors, bayes):
        class_prob = hmm_val/bayes_val + cnn_val/bayes_val
        p_models.append(class_prob)
    return p_models

if __name__ == "__main__":
    cb = calculate_bayes(hmm, cnn)
    print(cb)

