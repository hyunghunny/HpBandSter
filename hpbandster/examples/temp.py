import csv

from lookup import *
from match import SurrogateMatcher

lookup = load('data2', data_folder='./lookup/')
print(len(lookup.get_hyperparam_vectors()))
print(lookup.begin_index)
lookup.begin_index += 1
print(lookup.begin_index)