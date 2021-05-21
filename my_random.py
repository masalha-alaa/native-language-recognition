"""
Custom random class
"""

import random


SEED = 42


def sample(population, k):
    random.seed(SEED)
    return random.sample(population, k)
