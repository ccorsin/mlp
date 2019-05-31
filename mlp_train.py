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
from random import random

class Train:
    def __init__(self, data, epochs, lr, visu, batch):
        self.data = data
        self.lr = lr
        self.visu = visu
        self.epochs = epochs
        self.batch_size = batch
        self.params = []
        self.costs = {}

    def train(self):
        # sns.set(style="ticks", color_codes=True)
        # sns.pairplot(self.data, hue = 'M')
        # plt.tight_layout()
        # plt.savefig('pair_plot_selected.pdf')
        Network([23, 6, 4, 2]).gardient_descent(self.data, self.epochs, self.lr, self.batch_size)

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.network = list()
        i = 1
        while i < self.num_layers:
            layer = {'W' : np.random.randn(self.sizes[i], self.sizes[i - 1]), 'b' : np.zeros(shape=(self.sizes[i], 1))}
            # layer = [{'weights':[random() for j in range(self.sizes[i - 1] + 1)]} for j in range(self.sizes[i])]
            self.network.append(layer)
            i += 1
        # print (self.network)

    def ft_get_stats(self, matrix):
        minmax = list()
        for column in matrix:
            stats = [matrix[column].min(), matrix[column].max()]
            minmax.append(stats)
        return minmax

    def normalize_dataset(self, dataset, minmax):
        normalized_data = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        return normalized_data

    def ft_standardize(self, matrix):
        print (matrix)
        return (matrix - matrix.mean()) / matrix.std()

    def build_excpected(self, data):
        expected = []
        for row in data.iterrows():
            index, batch = row
            inputs = batch.tolist()
            if inputs[0] == 'M':
                expected.append([0, 1])
            else:
                expected.append([1, 0])
        df_ex = pd.DataFrame(expected)
        return df_ex

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.column_stack((e_x.sum(axis=1),e_x.sum(axis=1)))

    def forward_propagation(self, data):
        inputs = data
        caches = []
        i = 0
        for layer in self.network:
            Z = np.dot(inputs, layer['W'].T) + layer['b'].T
            if i < len(self.network) - 1:
                layer['A'] = self.sigmoid(Z)
            else:
                layer['A'] = self.softmax(Z)
            layer['A_prev'] = inputs
            inputs = layer['A']
            cache = (inputs, layer['W'], layer['b'], layer['A_prev'])
            caches.append(cache)
            i += 1
        return inputs, caches

    def ft_cost_evaluation(self, output, y):
        m = y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(y, np.log(output)) + np.multiply(1 - y, np.log(1 - output)), axis=None)        
        return cost

    def backward_propagation(self, output, y, caches):
        gradients = []
        m = y.shape[1]
        dA = - (np.divide(y, output) - np.divide(1 - y, 1 - output))
        cache = caches[-1]
        dZ = dA * self.sigmoid_prime(cache[0])
        grads = {}
        grads['dW'] = np.dot(dZ.T, cache[3]) / m
        grads['db'] = np.sum(np.squeeze(np.sum(dZ, axis=1))) / m
        dA_previous = np.dot(dZ, cache[1])
        gradients.append(grads)
        for i in reversed(range(len(self.network) - 1)):
            grads = {}
            cache = caches[i]
            dZ = dA_previous * self.sigmoid_prime(cache[0])
            grads['dW'] = np.dot(dZ.T, cache[3]) / m
            grads['db'] = np.sum(np.squeeze(np.sum(dZ, axis=1))) / m
            dA_previous =  np.dot(dZ, cache[1])
            gradients.insert(0, grads)
        return gradients

    def update_weights(self, data, lr, gradient):
        for i in range(len(self.network)):
            self.network[i]['W'] = self.network[i]['W'] - lr * gradient[i]['dW']
            self.network[i]['b'] = self.network[i]['b'] - lr * gradient[i]['db']

    def predict(self, inputs):
        outputs = self.forward_propagation(inputs)
        return outputs

    def evaluate_perfo(self, prediction, y):
        error = np.sum((np.argmax(prediction, axis=1) - np.argmax(y, axis=1)) ** 2)
        acc = (len(y) - error) * 100 / len(y)
        print (error, len(y))
        return acc

    def gardient_descent(self, data, epochs, lr, batch_size):
        y = self.build_excpected(data)
        data = data.iloc[:, 1:]
        minmax = self.ft_get_stats(data)
        std_data = self.normalize_dataset(data, minmax)
        for epoch in range(epochs):
            outputs, caches = self.forward_propagation(std_data)
            cost = self.ft_cost_evaluation(outputs, y)
            gradients = self.backward_propagation(outputs, y, caches)
            self.update_weights(std_data, lr, gradients)
            print('>epoch=%d, lrate=%.3f, cost=%d' % (epoch, lr, np.sum(cost)))
        df = pd.read_csv('data_test.csv', sep=',')
        df = df.dropna()
        df = df.iloc[:, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]]
        y_test = self.build_excpected(df).values
        data_test = df.iloc[:, 1:]
        std_data_test = self.normalize_dataset(data_test, minmax)
        prediction, _ = self.predict(std_data_test)
        accuracy = self.evaluate_perfo(prediction, y_test)
        print (accuracy)

args = argparse.ArgumentParser("Statistic description of your data file")
args.add_argument("file", help="File to descripte", type=str)
args.add_argument("-e", "--epoch", help="The number of iterations to go through the regression", default=1000, type=int)
args.add_argument("-l", "--learning", help="The learning rate to use during the regression", default=0.005, type=float)
args.add_argument("-v", "--visu", help="Visualize functions", action="store_true", default=False)
args.add_argument("-b", "--batch", help="Adjust batch size", default=10, type=int)
args = args.parse_args()

if os.path.isfile(args.file):
    try:
        df = pd.read_csv(args.file, sep=',')
        df = df.dropna()
        df = df.iloc[:, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]]
        Train(df, args.epoch, args.learning, args.visu, args.batch).train()
        
    except Exception as e:
        raise(e)
        sys.stderr.write(str(e) + '\n')
        sys.exit()
else:
    sys.stderr.write("Invalid input\n")
    sys.exit(1)