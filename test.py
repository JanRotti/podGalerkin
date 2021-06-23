import numpy as np

def test_func(x):
        global factor
        print(factor*x)

factor = 1

test_func(2)