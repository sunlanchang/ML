from sklearn import preprocessing
import numpy as np

a = np.array([[1, 2, 3], [111, 222, 333], [
             1111, 2222, 3333]], dtype=np.float64)
print(a)
print(preprocessing.scale(a))
