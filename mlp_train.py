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
        Network([22, 6, 4, 2]).gardient_descent(self.data, self.epochs, self.lr, self.batch_size)

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.network = list()
        i = 1
        while i < self.num_layers:
            layer = [{'weights':[random() for j in range(self.sizes[i - 1] + 1)]} for j in range(self.sizes[i])]
            self.network.append(layer)
            i += 1

    def ft_get_stats(self, matrix):
        minmax = list()
        for column in matrix:
            stats = [matrix[column].min(), matrix[column].max()]
            minmax.append(stats)
        return minmax

    def normalize_dataset(self, dataset, minmax):
        normalized_data = (dataset-dataset.min()) / (dataset.max() - dataset.min())
        return normalized_data
        # i = 0
        # for elem in dataset:
        #     std_elem = (elem - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        #     data.append(std_elem)
        #     i += 1
        # return data

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


    def activate(self, weights, inputs):
        b = weights[-1]
        a = 0.0
        for i in range(len(weights) - 1):
            a += weights[i] * inputs[i]
        return a + b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def forward_propagation(self, data):
        inputs = data
        l = 0
        for layer in self.network:
            l += 1
            new_inputs = []
            for neuron in layer:
                z = np.dot(inputs, neuron['weights'])
                # z = self.activate(neuron['weights'], inputs)
                neuron['a'] = self.sigmoid(z)
                new_inputs.append(neuron['a'])
            if l < len(self.network):
                new_inputs.append(np.ones(len(inputs)))
            inputs = pd.DataFrame(new_inputs).T
        return inputs

    def ft_error_evaluation(self, output, y):
        cost = (output - y) ** 2
        return cost.sum(axis=1).sum(axis=0)

    def backward_propagation(self, expected):
        # db = [np.zeros(b.shape) for b in self.biases]
        # dw = [np.zeros(w.shape) for w in self.weights]
        # activation = x
        # activations = [x]
        # zs = []
        # for b, w in zip(self.biases, self.weights):
        #     z = np.dot(w, activation) + b
        #     zs.append(z)
        #     activation = self.sigmoid(z)
        #     activations.append(activation)
        # error = activations[-1] - y
        # delta = error * self.sigmoid_prime(zs[-1])
        # db[-1] = d
        # dw[-1] = np.dot(d, activations[-2].T)
        # for l in range(2, self.num_layers):
        #     z = zs[-1]
        #     d = np.dot(self.weights[-l + 1].T, d) * self.sigmoid_prime(z)
        #     db[-1] = d
        #     dw[-1] = np.dot(d, activations[-l - 1].T)
        # return (db, dw)
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = float(0)
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['a'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.sigmoid_prime(neuron['a'])

    def update_weights(self, data, lr):
        # for i in range(len(self.network)):
        #     inputs = data[:-1]
        #     if i != 0:
        #         inputs = [neuron['a'] for neuron in self.network[i - 1]]
        #     for neuron in self.network[i]:
        #         for j in range(len(inputs.columns)):
        #             neuron['weights'][j] += lr * neuron['delta'] * inputs[j]
        #         neuron['weights'][-1] += lr * neuron['delta']
        for i in range(len(self.network)):
            inputs = data
            if i != 0:
                inputs = pd.DataFrame([neuron['a'] for neuron in self.network[i - 1]]).T
            for neuron in self.network[i]:
                for j in range(len(neuron['weights'])):
                    neuron['weights'][j] += (lr / len(neuron['a'])) * np.dot(neuron['delta'].T, inputs).sum(axis=0)
                neuron['weights'][-1] += lr * neuron['delta'].sum(axis=0)

    def predict(self, row):
        outputs = self.forward_propagation(row)
        return outputs

    def gardient_descent(self, data, epochs, lr, batch_size):
        n = len(data)
        # y = data.iloc[:, 1:2]
        # row = [1, 0]
        # exepected = [0, 1]
        # output = self.forward_propagation(row)
        # self.backward_propagation(exepected)
        # for layer in self.network:
        #     print(layer)
        # print (output)
        y = self.build_excpected(data)
        sum_err = 0
        data = data.iloc[:, 1:]
        minmax = self.ft_get_stats(data)
        std_data = self.normalize_dataset(data, minmax)
        for epoch in range(epochs):
            outputs = self.forward_propagation(std_data)
            error = self.ft_error_evaluation(outputs, y)
            self.backward_propagation(y)
            self.update_weights(std_data, lr)
            # for row in data.iterrows():
            #     index, batch = row
            #     inputs = batch.tolist()
            #     if inputs[0] == 'M':
            #         expected = [0, 1]
            #     else:
            #         expected = [1, 0]
            #     del inputs[0]
            #     std_inputs = self.normalize_dataset(inputs, minmax)
            #     outputs = self.forward_propagation(std_inputs)
            #     sum_err += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            #     self.backward_propagation(expected)
            #     self.update_weights(inputs, lr)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lr, error))
        # t = 0
        # f = 0
        # df = pd.read_csv('data_test.csv', sep=',')
        # df = df.dropna()
        # df = df.iloc[:, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]]
        # for row in df.iterrows():
        #     index, batch = row
        #     test = batch.tolist()
        #     if test[0] == 'M':
        #         expected = [0, 1]
        #     else:
        #         expected = [1, 0]
        #     del test[0]
        #     std_test = self.normalize_dataset(test, minmax)
        #     prediction = self.predict(std_test)
        #     if (max(prediction) == prediction[0] and expected[0] == 1) or (max(prediction) == prediction[1] and expected[1] == 1):
        #         t += 1
        #     else:
        #         f += 1
        # print ('Accuracy :', t * 100 / (t + f))

        # for j in range(epochs):
        #     # np.random.shuffle(data)
        #     batches = []
        #     for k in range(0, n, batch_size):
        #         batches.append(data[k:k+batch_size])
        #     for batch in batches:
        #         # iterate over training examples of batch
        #         exit ()
        #     print ("Epoch ", j)
        #     # self.forward_propagation(data)
        #     self.backward_propagation(data, y)


args = argparse.ArgumentParser("Statistic description of your data file")
args.add_argument("file", help="File to descripte", type=str)
args.add_argument("-e", "--epoch", help="The number of iterations to go through the regression", default=100, type=int)
args.add_argument("-l", "--learning", help="The learning rate to use during the regression", default=0.01, type=float)
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