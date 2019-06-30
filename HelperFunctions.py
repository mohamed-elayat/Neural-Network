
import math
import numpy as np
from numba import vectorize, float64, cuda

@vectorize([float64(float64)])
def plusOne(x):
    return(x+1)

@vectorize([float64(float64)])  #todo: add cuda thing
def sigmoid(z):
    return (  1 + math.exp(-z)  ) ** -1

@vectorize ([float64(float64)])
def sigmoidPrime(z):
    sig = (  1 + math.exp(-z)  ) ** -1
    return sig * (1-sig)

# @vectorize(['float64(float64, float64, float64)'], target='cuda')
# def coefficient_adjustement(learning_rate_with_1_over_m: float, coefficient: float, big_delta: float):
#     return coefficient - (learning_rate_with_1_over_m * big_delta)


@vectorize(  [float64(float64, float64)]  )
def costFunction(  prediction, actual  ):

    one = -actual * math.log10(  prediction  )
    two = -(1-actual) * math.log10(  1 - prediction  )

    return one + two


@vectorize (  [float64(float64, float64, float64)]  )
def updateWeights(  oldWeight, gradient, alpha  ):
    return oldWeight - alpha * gradient


omak2 = np.array(  ([1, 2, 19, 1], [3, 4, 32, 3], [5, 6, 21, 3])  )
omak4 = np.array(  ([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])  )
# print(  np.mean(  np.sum(omak2, axis=1)  )  )

# print(  np.mean(  np.sum(  omak2, axis=1  )  )  )

# np.delete(arr, 1, 0)

omak3 = np.multiply(  omak2, 5  )
omak3 = np.divide(  omak3, 5  )
# print(  np.delete(  omak2, 3, 1  )  )
print(  omak3  )

# a = np.array(  ([0,1,2], [3,4,5], [6,7,8])  )
#
# print(  sigmoid(a)  )
# print (  sigmoidPrime(a)  )

# omak = [None] * 5
# # omak.append("koss o5tak")
#
# omak[3] = "koss o5tak"
#
# print(omak)
