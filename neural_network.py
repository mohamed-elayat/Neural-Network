

import numpy as np
from helper_functions import sigmoid, sigmoidPrime, matrixCostFunction, matrixUpdateWeights
from timeit import default_timer as timer


class NeuralNetwork:

    def __init__(self, x: np.ndarray, y: np.ndarray, layers: int, nodes: int):

        self.x = x
        self.y = y
        self.yOneHot = self.oneHot(self.y)      #we need our y in one hot to calculate the cost function/gradients

        self.m = self.x.shape[0]                #number of examples in our dataset
        self.features = self.x.shape[1]         #number of features
        self.classes = self.yOneHot.shape[1]    #number of output classes
        self.layers = layers                    #total number of layers
        self.nodes = nodes                      #number of nodes each hidden layer

        self.printHyperparameters()

        self.weights = []
        self.initWeights()


# /******************************************************************************************************
#  * Main functions
#  ******************************************************************************************************/

    def train(self, iterations, learningRate):

        print("Iteration count: " + str(iterations) + "\n" + "Learning rate: " + str(learningRate) + "\n")
        start = timer()

        for i in range(iterations):
            self.forwardPropagate()
            print(str(i) + ":     " + str(self.costFunction()))
            self.backwardPropagate()
            self.updateWeights(learningRate)

        duration = timer() - start
        print("\nTotal computation time: " + str(duration) + " seconds \n")


    #For each layer starting layer 2, we calculate z, a and aPrime
    #for all of the layer's nodes.
    def forwardPropagate(self):

        self.resetValues()
        ones = np.ones(  (self.m, 1)  )

        for i in range(  self.layers-1  ):

            self.a[i] = np.hstack(  (ones, self.a[i])  )        #Ones column is added to be multiplied by the biases

            self.z.append(  self.a[i] @ self.weights[i]  )
            self.a.append(  sigmoid(self.z[i+1])  )
            self.aPrimes.append(  sigmoidPrime(self.a[i+1])  )

        return self.a[self.layers-1]


    #First find the deltas for each layer's nodes,
    #then find the gradients for the weights
    def backwardPropagate(self):

        self.deltas = [None] * self.layers
        self.deltas[self.layers-1] = self.a[self.layers-1] - self.yOneHot

        for i in range(  self.layers-2, 0, -1  ):
            weightTranspose = np.transpose(self.weights[i])

            almostDelta = self.deltas[i+1] @ weightTranspose
            almostDelta = np.delete(  almostDelta, 0, 1  )

            self.deltas[i] = np.multiply(  almostDelta, self.aPrimes[i]  )

        for i in range(  self.layers-1  ):
            aTranspose = np.transpose(  self.a[i]  )
            self.gradients.append(  aTranspose @ self.deltas[i+1]  )
            self.gradients[i] = np.divide(  self.gradients[i], self.m  )


    def costFunction(self):

        costMatrix = matrixCostFunction(self.a[self.layers - 1], self.yOneHot)
        return np.mean(  costMatrix  )


    def updateWeights(self, learningRate):

        for i in range(  self.layers-1  ):
            alphaMatrix = np.full(  self.gradients[i].shape, learningRate  )
            self.weights[i] = matrixUpdateWeights(self.weights[i], self.gradients[i], alphaMatrix)


# /******************************************************************************************************
#  * Secondary functions
#  ******************************************************************************************************/

    def oneHot(self, y):

        distinct = np.unique(y)                      #Method to obtain the number of output classes.
        numberOfClasses = distinct.shape[0]          #todo: implement a more reliable method for number of output classes

        y = y.astype('int32')

        yOneHot = np.zeros(  (y.shape[0], numberOfClasses)  )
        yOneHot[np.arange(y.shape[0]), y-1] = 1
        return yOneHot

    # Create our weight matrices and assign them random valuables in [-1,1]
    def initWeights(self):

        self.weights.append(  np.array(np.random.random_sample((self.features+1, self.nodes)) * 2 - 1, dtype=np.float)  )

        for i in range(1, self.layers-2):
            self.weights.append(  np.array(np.random.random_sample((self.nodes + 1, self.nodes)) * 2 - 1, dtype=np.float)  )

        self.weights.append(  np.array(np.random.random_sample((self.nodes + 1, self.classes)) * 2 - 1, dtype=np.float)  )

    # Reset the layer values and the gradients before forward propagation
    def resetValues(self):

        self.a = []
        self.aPrimes = []
        self.z = []
        self.deltas = []
        self.gradients = []

        #Only the input is needed for the first layer.
        #Something is added to z and aPrime to align the 3 lists.
        self.z.append(  np.zeros(  self.x.shape  )  )
        self.a.append(self.x)
        self.aPrimes.append(  np.zeros(  self.x.shape  )  )


    def showResults(self):

        prediction = self.forwardPropagate()

        wins = 0
        losses = 0

        for indexRow, row in enumerate(prediction):

            max = 0
            max_idx = 0

            for indexCol, col in enumerate(row):
                if col > max:
                    max = col
                    max_idx = indexCol

            # print("max: " + str(max) + " and index: " + str(max_idx + 1))

            if max_idx + 1 == self.y[indexRow]:
                wins = wins + 1
            else:
                losses = losses + 1

        print("right: " + str(wins))
        print("wrong: " + str(losses))


    def printHyperparameters(self):

        print("x shape: " + str(self.x.shape))
        print("y shape: " + str(self.yOneHot.shape))
        print("Layers: " + str(self.layers))
        print("Nodes in hidden layers: " + str(self.nodes))
        print("Features: " + str(self.features))
        print("Classes: " + str(self.classes))
        print("Examples: " + str(self.m) + "\n")