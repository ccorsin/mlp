import sys
import math
import csv
import os
import argparse
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

class Train:
    def __init__(self, data, epochs, lr, visu):
        self.data = data
        self.lr = lr
        self.visu = visu
        self.epochs = epochs
        self.params = []
        self.costs = {}

    def train(self):
        # sns.set(style="ticks", color_codes=True)
        # sns.pairplot(self.data, hue = 'M')
        # plt.tight_layout()
        # plt.savefig('pair_plot_selected.pdf')
        Network([2, 3, 2]).gardient_descent(self.data, self.epochs, self.lr)


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_propagation(self, a):
        for b, w in zip(self.biases, self.weights):
            print (len(w), len(a))
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def backward_propagation(self, x, y):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        error = activations[-1] - y
        delta = error * self.sigmoid_prime(zs[-1])
        db[-1] = d
        dw[-1] = np.dot(d, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-1]
            d = np.dot(self.weights[-l + 1].T, d) * self.sigmoid_prime(z)
            db[-1] = d
            dw[-1] = np.dot(d, activations[-l - 1].T)
        return (db, dw)

    # def cost_evaluation(p, y):
    #     np.sum((p - y) * (p - y))


    def gardient_descent(self, data, epochs, lr):
        n = len(data)
        y = data.iloc[:, 1:2]
        for j in range(epochs):
            print ("Epoch ", j)
            self.forward_propagation(data)
            self.backward_propagation(data, y)



args = argparse.ArgumentParser("Statistic description of your data file")
args.add_argument("file", help="File to descripte", type=str)
args.add_argument("-e", "--epoch", help="The number of iterations to go through the regression", default=1000, type=int)
args.add_argument("-l", "--learning", help="The learning rate to use during the regression", default=0.01, type=float)
args.add_argument("-v", "--visu", help="Visualize functions", action="store_true", default=False)
args = args.parse_args()

if os.path.isfile(args.file):
    try:
        df = pd.read_csv(args.file, sep=',')
        df = df.dropna()
        df = df.iloc[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]]
        Train(df, args.epoch, args.learning, args.visu).train()
        
    except Exception as e:
        raise(e)
        sys.stderr.write(str(e) + '\n')
        sys.exit()
else:
    sys.stderr.write("Invalid input\n")
    sys.exit(1)