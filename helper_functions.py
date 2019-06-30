
import math
from numba import vectorize, float64, cuda


#The vectorize decorator allows a function that takes
#scalar inputs and outputs to take matrices instead. The
#operations are performed element-wise
@vectorize(  [float64(float64)]  )
def sigmoid(z):
    return (  1 + math.exp(-z)  ) ** -1

@vectorize (  [float64(float64)]  )
def sigmoidPrime(z):
    sig = (  1 + math.exp(-z)  ) ** -1
    return sig * (1-sig)

@vectorize(  [float64(float64, float64)]  )
def matrixCostFunction(prediction, actual):
    one = -actual * math.log10(  prediction  )
    two = -(  1-actual  ) * math.log10(  1 - prediction  )
    return one + two

@vectorize (  [float64(float64, float64, float64)]  )
def matrixUpdateWeights(oldWeight, gradient, alpha):
    return oldWeight - alpha * gradient

