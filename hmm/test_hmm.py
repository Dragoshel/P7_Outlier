from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from numpy import array, ones, tile, exp

# ------------------------------------------------------
# Define data
# ------------------------------------------------------
zero = [0, 1, 1, 0,
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 0]

one = [0, 1, 1, 0,
       0, 0, 1, 0,
       0, 0, 1, 0,
       0, 0, 1, 0]

two = [0, 1, 1, 0,
       0, 0, 1, 0,
       0, 1, 0, 0,
       0, 1, 1, 0]

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
p_zero = prior_no(0, zero)
e_zero = [1-sum(zero)/16, sum(zero)/16]
r_one = split_no(one)
p_one = prior_no(1, one)
e_one = [1-sum(one)/16, sum(one)/16]
r_two = split_no(two)
p_two = prior_no(2, two)
e_two = [1-sum(two)/16, sum(two)/16]

fit_sample = array([r_zero, r_one, r_two, r_zero, r_one, r_two])
fit_priors = array([p_zero, p_one, p_two, p_zero, p_one, p_two])
print(fit_sample.shape)
print(fit_priors.shape)

test_sample = array([r_zero, r_one, r_two])
test_priors = array([p_zero, p_one, p_two])

# ------------------------------------------------------
# Define model
# ------------------------------------------------------
# Time t              => each field in the array
# Hidden state        => zero, one, two
# Observation records => 0 or 1
model = DenseHMM(max_iter=10)

# Distributions, where one is uniform
# and one is preferable towards ones
zero_d = Categorical([e_zero])
one_d = Categorical([e_one])
two_d = Categorical([e_two])
model.add_distributions([zero_d, one_d, two_d])

# Defining equal start probabilities
model.add_edge(model.start, zero_d, 0.33)
model.add_edge(model.start, one_d, 0.33)
model.add_edge(model.start, two_d, 0.33)

# Defining edges between the distributions
model.add_edge(zero_d, zero_d, 0.4)
model.add_edge(zero_d, one_d, 0.3)
model.add_edge(zero_d, two_d, 0.3)
model.add_edge(one_d, zero_d, 0.3)
model.add_edge(one_d, one_d, 0.4)
model.add_edge(one_d, two_d, 0.3)
model.add_edge(two_d, zero_d, 0.3)
model.add_edge(two_d, one_d, 0.3)
model.add_edge(two_d, two_d, 0.4)

# Define end states
#model.add_edge(zero_d, model.end, 0.33)
#model.add_edge(one_d, model.end, 0.33)
#model.add_edge(uniform_d, model.end, 0.33)

# Train model
model.fit(fit_sample, priors=fit_priors)

# ------------------------------------------------------
# Predict classes
# ------------------------------------------------------
def print_all_predictions(all_preds, all_probs):
    for preds, probs in zip(all_preds, all_probs):
        for pred, prob in zip(preds, probs):
            print('Probability: {}, Prediction: {}'.format(str(prob), pred.item()))
        print(' ')

def print_predicted_label(all_preds):
    for preds in all_preds:
        print('Predicted label: {}'.format(preds[-1]))

all_preds = model.predict(test_sample, priors=test_priors)
all_probs = model.predict_proba(test_sample, priors=test_priors)

print_all_predictions(all_preds, all_probs)
#print_predicted_label(all_preds)

print('Probabilities:')
print(exp(model.log_probability(test_sample)))
#print(model.distributions.log_prob(test_sample))
