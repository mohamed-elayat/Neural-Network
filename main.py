import argparse
import numpy as np
import sys
from neural_network import NeuralNetwork    #todo: use cuda to compute faster.

np.set_printoptions(threshold=sys.maxsize)  #for testing purposes

#Argument parser to set the hyperparameters for the NN
parser = argparse.ArgumentParser(description='Enter NN hyperparameters')
parser.add_argument("--layers",            type=int,   default=4,       help="Number of layers")
parser.add_argument("--nodes",             type=int,   default=256,     help="Number of nodes in hidden layers")
parser.add_argument("--iterationCount",    type=int,   default=150,     help="Number of gradient descents")
parser.add_argument("--learningRate",      type=float, default=0.7,     help="Gradient descent learning rate", choices=np.linspace(0, 1, 101))
parser.add_argument("--cudaDeviceID",      type=int,   default=0,       help="Cuda device to use")
parser.add_argument("--XPath",             type=str,   default=r".\data\digits.csv")
parser.add_argument("--YPath",             type=str,   default=r".\data\labels.csv")
parser.add_argument("--savePath",          type=str,   default=r".\data\results.csv")


def main():

    args = parser.parse_args()

    #Load the inputs and outputs into numpy matrices
    x = np.loadtxt(args.XPath, delimiter=",")
    y = np.loadtxt(args.YPath, delimiter=",")

    NN1 = NeuralNetwork(  x, y, args.layers, args.nodes  )
    NN1.train(  args.iterationCount, args.learningRate  )
    NN1.showResults()

main()

