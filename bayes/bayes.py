from pomegranate.distributions import Categorical
from pomegranate.bayes_classifier import BayesClassifier

NO_CLASSES = 10

dist = Categorical([0.50, 0.50])
model = BayesClassifier([dist]*NO_CLASSES)

