
"""Minimal numpy fallback for basic operations."""
import math
import random

class ndarray:
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
        else:
            self.data = [data]
        self.dtype = dtype
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, key):
        return self.data[key]
        
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def mean(self):
        return sum(self.data) / len(self.data)
        
    def sum(self):
        return sum(self.data)

def array(data, dtype=None):
    return ndarray(data, dtype)

def zeros(shape):
    if isinstance(shape, int):
        return ndarray([0] * shape)
    else:
        return ndarray([0] * shape[0])

def ones(shape):
    if isinstance(shape, int):
        return ndarray([1] * shape)
    else:
        return ndarray([1] * shape[0])

def random_normal(size):
    return ndarray([random.gauss(0, 1) for _ in range(size)])

# Basic constants
pi = math.pi
e = math.e
