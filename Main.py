import argparse
import numpy as np
import sys

from HelperFunctions import sigmoid, sigmoidPrime, costFunction, updateWeights

from numba import vectorize, cuda
from timeit import default_timer as timer
np.set_printoptions(threshold=sys.maxsize)


parser = argparse.ArgumentParser(description='Enter NN hyperparameters')
parser.add_argument("--layers",      type=int,   default=4,      help="Number of layers")
parser.add_argument("--iterationCount",    type=int,   default=50,     help="Number of gradient descents")
parser.add_argument("--learningRate",      type=float, default=0.01,   help="Gradient descent learning rate", choices=np.linspace(0, 1, 101))
parser.add_argument("--XPath",             type=str,   default=r".\data\digits.csv")
parser.add_argument("--YPath",             type=str,   default=r".\data\labels.csv")
parser.add_argument("--savePath",          type=str,   default=r".\data\results.csv")
parser.add_argument("--cudaDeviceID",     type=int,   default=0,      help="Cuda device to use")



class NeuralNetwork:

    def __init__(self, x: np.ndarray, y: np.ndarray, layers: int):

        print(x.shape)
        print(y.shape)

        self.x = x

        temp = y.astype('int32')
        self.y = np.zeros((x.shape[0], 10))
        self.y[np.arange(x.shape[0]), temp - 1] = 1

        self.m = self.x.shape[0]
        self.features = self.x.shape[1]
        self.classes = self.y.shape[1]
        self.layers = layers

        self.hiddenLayerNodes = 256 #todo: parametrize this.

        self.weights = []
        self.initWeights()


    def initWeights(self):

        self.weights.append(np.array(  np.random.random_sample(  (self.features+1, self.hiddenLayerNodes)  ) * 2 - 1, dtype=np.float  ))

        for i in range(1, self.layers-2):
            self.weights.append(np.array(  np.random.random_sample(  (self.hiddenLayerNodes+1, self.hiddenLayerNodes)  ) * 2 - 1, dtype=np.float  ))

        self.weights.append(np.array(  np.random.random_sample(  (self.hiddenLayerNodes + 1, self.classes)  ) * 2 - 1, dtype=np.float  ))

    def resetValues(self):

        self.a = []
        self.aPrimes = []
        self.z = []
        self.deltas = []
        self.gradients = []

        self.z.append(  np.zeros(  self.x.shape  )  )
        self.a.append(self.x)
        self.aPrimes.append(  np.zeros(  self.x.shape  )  )


    def forwardPropagate(self):

        self.resetValues()

        ones = np.ones(  (self.m, 1)  )

        for i in range(  self.layers-1  ):

            self.a[i] = np.hstack(  (ones, self.a[i])  )

            self.z.append(  self.a[i] @ self.weights[i]  )
            self.a.append(  sigmoid(self.z[i+1])  )
            self.aPrimes.append(  sigmoidPrime(self.a[i+1])  )

        return self.a[self.layers-1]

    def backwardPropagate(self):

        self.deltas = [None] * self.layers
        self.deltas[self.layers-1] = self.a[self.layers-1] - self.y

        # range(10, -11, -1)

        for i in range(  self.layers-2, 0, -1):
            temp = np.transpose(self.weights[i])

            temp2 = self.deltas[i+1] @ temp
            temp2 = np.delete(  temp2, 0, 1  )

            self.deltas[i] = np.multiply(  temp2, self.aPrimes[i]  ) #todo: eliminate the first column of aPrime and deltas.Will make it faster.

        for i in range(  self.layers-1  ):
            # temp1 = np.delete(  self.deltas[i+1], 0, 1  )
            temp2 = np.transpose(  self.a[i]  )
            self.gradients.append(  temp2 @ self.deltas[i+1]  )
            self.gradients[i] = np.divide(  self.gradients[i], self.m  )




    def costFunction(self):

        temp = costFunction(  self.a[self.layers-1], self.y  )
        temp2 = np.sum(  temp, axis=1  )
        # return np.mean(  temp2  )
        return np.mean(temp)

    def costFunction2(self):

        temp = costFunction(  self.a[self.layers-1], self.y  )
        temp2 = np.sum(  temp, axis=1  )
        # return np.mean(  temp2  )
        return np.mean(temp2)


    def updateWeights(self):

        for i in range(self.layers-1):

            mMatrix = np.full(self.gradients[i].shape, self.m)
            alphaMatrix = np.full(self.gradients[i].shape, 0.7)
            self.weights[i] = updateWeights(  self.weights[i], self.gradients[i], alphaMatrix  )





def main():

    args = parser.parse_args()
    print("Layers: " + str(args.layers)  )
    print("Iteration count: " + str(args.iterationCount)  )
    print("Learning rate: " + str(args.learningRate)  )
    print("X path: " + args.XPath)
    print("Y path: " + args.YPath)
    print("Save path: " + args.savePath)
    print("Cuda device: " + str(args.cudaDeviceID))

    x = np.loadtxt(args.XPath, delimiter=",")
    y = np.loadtxt(args.YPath, delimiter=",")

    yaRab = y

    layers = args.layers

    NN1 = NeuralNetwork(x, y, layers)

    print("Number of examples = " + str(NN1.m))
    print("Number of layers = " + str(NN1.layers))
    print("Number of features = " + str(NN1.features))
    print("Number of classes = " + str(NN1.classes))

    # print(  NN1.forwardPropagate()  )

    NN1.forwardPropagate()
    print(NN1.costFunction())
    NN1.backwardPropagate()
    print(NN1.gradients[NN1.layers - 2])


    for i in range(3500):
        NN1.forwardPropagate()
        print(  str(NN1.costFunction()) + "    " +  str(NN1.costFunction2()) + "    " +  str(i) )
        NN1.backwardPropagate()
        NN1.updateWeights()

    print(NN1.gradients[NN1.layers-2])
    print(NN1.weights[NN1.layers-2])
    # print(NN1.a[NN1.layers-1])

    koss = NN1.a[NN1.layers-1]

    wins = 0
    losses = 0

    # for row in koss:
    for indexRow, row in enumerate(koss):

        max = 0
        max_idx = 0

        for index, col in enumerate(row):
            if col > max:
                max = col
                max_idx = index

        print("max: " + str(max) + " and index: " + str(max_idx+1) )

        if max_idx+1 == yaRab[indexRow]:
            wins = wins + 1
        else:
            losses = losses + 1

    print(  " right: " + str(wins)  )
    print(  " wrong: " + str(losses))




main()
# justPlaying()


#todo:#########################################################################################################################
#todo:#########################################################################################################################
#todo:#########################################################################################################################
#todo:#########################################################################################################################
#todo:#########################################################################################################################
#todo:#########################################################################################################################


# def justPlaying():
#
#
#     # ones = np.ones((m, 1))  # Generate a one filled matrix with 1 col and m rows for the bias
#     # a = np.hstack((ones, x))  # Append the input matrix after the ones matrix
#     # z = a @ coef  # Compute the results of the input X coefficients
#
#     ones = np.ones(  (5, 1)  )
#     temp = np.array(  ([2],[3],[4],[5],[6])  )
#
#     temp2 = np.array(  [1, 1, 1, 1, 1])
#
#     tot = np.hstack(  (ones,temp)  )
#
#     print(temp2 @ tot)




# class Person:
#   def __init__(self, name, age):
#     self.name = name
#     self.age = age
#
# p1 = Person("John", 36)
#
# print(p1.name)
# print(p1.age)


# print(x)
# print(y)


# print(y.shape)
# print(x.shape)
# print(x[0, 155])
# print(x.shape[1])