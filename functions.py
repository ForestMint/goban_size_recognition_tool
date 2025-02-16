## ------------------- IMPORT PACKAGES ----------------------------------------

import numpy as np

## ------------------- DECLARE FUNCTIONS -------------------------------

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
