from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from numpy import array, ones, tile, exp, full

# ------------------------------------------------------
# Define data
# ------------------------------------------------------
zero = [0, 3, 2, 0,
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 0]

one = [0, 1, 1, 0,
       0, 0, 1, 0,
       0, 0, 1, 0,
       0, 0, 1, 0]

one = [2, 1, 1, 0,
       0, 0, 1, 0,
       0, 0, 1, 0,
       0, 2, 1, 2]

two = [1, 1, 1, 1,
       0, 0, 1, 0,
       0, 1, 0, 0,
       1, 1, 1, 1]

def split_no(number):
    return [[part] for part in number]

def prior_no(number, no_array):
    prior = ones(3) / 3
    prior = tile(prior, (16,1))
    for i in range(16):
        if no_array[i] == 1:
            prior[i] = [0,0,0]
            prior[i][number] = 1
    return prior

r_zero = split_no(zero)
r_one = split_no(one)
r_two = split_no(two)

fit_zeros = array([r_zero] * 10)
fit_ones = array([r_one] * 10)
fit_twos = array([r_two] * 10)
test_sample = array([r_zero, r_one, r_two])
test_labels = [0, 1, 2]

print("Shape of zero {}".format(fit_zeros.shape))

# ------------------------------------------------------
# Define model
# ------------------------------------------------------
# Time t              => each field in the array
# Hidden state        => zero, one, two
# Observation records => 0 or 1
models = []
fittings = [fit_zeros, fit_ones, fit_twos]

# num of dist
# perfferable half 0.8
# 

for digit in range(3):
    print('Training for digit {}'.format(digit))
    uniform_d = Categorical([[0.20, 0.20, 0.20, 0.20, 0.20]])
    ones_d = Categorical([[0.1, 0.1, 0.4, 0.4, 0.0]])
    model = DenseHMM([uniform_d, ones_d], max_iter=10)
    print(model)
    # Train model
#    model.fit(fittings[digit])
    models.append(model)

# ------------------------------------------------------
# Predict classes
# ------------------------------------------------------
# def print_all_predictions(all_preds, all_probs):
#     for preds, probs in zip(all_preds, all_probs):
#         for pred, prob in zip(preds, probs):
#             print('Probability: {}, Prediction: {}'.format(str(prob), pred.item()))
#         print(' ')

# def print_predicted_label(all_preds):
#     for preds in all_preds:
#         print('Predicted label: {}'.format(preds[-1]))

# pred_probs = []
# for idx, model in enumerate(models):
#     print('Testing digits on model {}'.format(idx))
#     all_preds = model.predict(test_sample)
#     all_probs = model.predict_proba(test_sample)
#     probs = model.log_probability(test_sample)
#     print(f'Predicted probs for model {idx}: {probs}')
#     probs = probs.tolist()
#     pred_probs.append(probs)
#     #print_all_predictions(all_preds, all_probs)
#     #print_predicted_label(all_preds)

# print('Probabilities:')
# pred_probs = array(pred_probs).transpose()
# all_probs = {label:probs for (label, probs) in zip(test_labels, pred_probs)}
# all_preds = [prob.argmax() for prob in pred_probs]
# for idx, pred in enumerate(all_preds):
#     print('Probabilities: {}, Prediction: {}, Actual: {}'.format(all_probs[idx], pred, idx))
