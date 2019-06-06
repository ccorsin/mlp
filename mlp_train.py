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
    def __init__(self, data, epochs, lr, visu):
        self.data = data
        self.lr = lr
        self.visu = visu
        self.epochs = epochs

    def train(self, net, details):
        # sns.set(style="ticks", color_codes=True)
        # sns.pairplot(self.data, hue = 'M')
        # plt.tight_layout()
        # plt.savefig('pair_plot_selected.pdf')
        network = Network([23, 6, 4, 2], net).gardient_descent(self.data, self.epochs, self.lr, self.visu, details)
        np.save('network.npy', network)

class Network:
    def __init__(self, sizes, network):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.network = list()
        i = 1
        if network == False:
            while i < self.num_layers:
                layer = {'W' : np.random.randn(self.sizes[i], self.sizes[i - 1]), 'b' : np.zeros(shape=(self.sizes[i], 1))}
                self.network.append(layer)
                i += 1
            np.save('initial_random_network.npy', self.network)
        else:
            self.network = np.load(network)
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

    def relu(self,x):
        return abs(x) * (x > 0)
    
    def relu_prime(self, x):
        return 1. * (x > 0)      

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.column_stack((e_x.sum(axis=1),e_x.sum(axis=1)))

    def softmax_prime(self, x):
        return x

    def forward_propagation(self, data):
        inputs = data
        caches = []
        i = 0
        for layer in self.network:
            Z = np.dot(inputs, layer['W'].T) + layer['b'].T
            layer['Z'] = Z
            if i < len(self.network) - 1:
                layer['A'] = self.relu(Z)
            else:
                layer['A'] = self.softmax(Z)
            layer['A_prev'] = inputs
            inputs = layer['A']
            cache = (layer['A'], layer['W'], layer['b'], layer['A_prev'], layer['Z'])
            caches.append(cache)
            i += 1
        return inputs, caches

    def ft_cost_evaluation(self, output, y):
        m = y.shape[0]
        output = output + 1e-15
        cost = (-1 / m) * np.sum((np.multiply(y, np.log(output)) + np.multiply(1 - y, np.log(1 - output)))[:,0])   
        return cost

    def ft_acc_evaluation(self, prediction, y):
        error = np.sum((np.argmax(prediction, axis=1) - np.argmax(y, axis=1)) ** 2)
        acc = (len(y) - error) / len(y)
        return acc

    def ft_precision_evaluation(self, prediction, y):
        tp = np.count_nonzero(np.argmax(prediction, axis=1) + np.argmax(y, axis=1) == 0)
        fp_tp = np.count_nonzero(np.argmax(prediction, axis=1) == 0)
        precision = tp / (fp_tp)
        return precision

    def backward_propagation(self, output, y, caches):
        gradients = []
        m = y.shape[0]
        dA = - (1 / m) * (np.divide(y, output) - np.divide(1 - y, 1 - output))
        cache = caches[-1]
        dZ = dA * self.softmax_prime(cache[0])
        grads = {}
        grads['dW'] = np.dot(dZ.T, cache[3]) / m
        grads['db'] = np.sum(np.squeeze(np.sum(dZ, axis=1))) / m
        dA_previous = np.dot(dZ, cache[1])
        gradients.append(grads)
        for i in reversed(range(len(self.network) - 1)):
            grads = {}
            cache = caches[i]
            dZ = dA_previous * self.relu_prime(cache[4])
            grads['dW'] = np.dot(dZ.T, cache[3]) / m
            grads['db'] = np.sum(np.squeeze(np.sum(dZ, axis=1))) / m
            dA_previous =  np.dot(dZ, cache[1])
            gradients.insert(0, grads)
        return gradients

    def update_weights(self, data, lr, gradient):
        for i in range(len(self.network)):
            self.network[i]['W'] = self.network[i]['W'] - lr * gradient[i]['dW']
            self.network[i]['b'] = self.network[i]['b'] - lr * gradient[i]['db']

    def gardient_descent(self, data, epochs, lr, visu, details):
        y = self.build_excpected(data)
        data = data.iloc[:, 1:]
        minmax = self.ft_get_stats(data)
        std_data = self.normalize_dataset(data, minmax)
        tr_y = y.values[:455, :]
        val_y = y.values[456:, :]
        tr_set = std_data.values[:455, :]
        val_set = std_data.values[456:, :]
        plot_tr_loss = []
        plot_val_loss = []
        plot_tr_acc = []
        val_cost = 0
        for epoch in range(epochs):
            tr_outputs, caches = self.forward_propagation(tr_set)
            cost = self.ft_cost_evaluation(tr_outputs, tr_y)
            gradients = self.backward_propagation(tr_outputs, tr_y, caches)
            self.update_weights(tr_set, lr, gradients)
            tr_acc = self.ft_acc_evaluation(tr_outputs, tr_y)
            val_outputs, _ = self.forward_propagation(val_set)
            val_cost_previous = val_cost
            val_cost = self.ft_cost_evaluation(val_outputs, val_y)
            tr_prec = self.ft_precision_evaluation(tr_outputs, tr_y)
            val_acc = self.ft_acc_evaluation(val_outputs, val_y)
            val_prec = self.ft_precision_evaluation(val_outputs, val_y)
            if val_cost > val_cost_previous and epoch:
                with open('minmax.json', 'w+') as json_file:  
                    json.dump(minmax, json_file)
                    if visu:
                        plt.plot(plot_tr_loss)
                        plt.plot(plot_tr_acc)
                        plt.plot(plot_val_loss)
                        plt.show()
                    np.save('network.npy', self.network)
                exit()
            if visu:
                plot_tr_loss.append(cost)
                plot_tr_acc.append(tr_acc)
                plot_val_loss.append(val_cost)
            if details:
                print('>epoch=%d, lrate=%.4f, loss=%.3f, acc=%.3f, prec=%.3f, val_loss=%.3f, val_acc=%.3f, val_prec=%.3f' % (epoch, lr, cost, tr_acc, tr_prec, val_cost, val_acc, val_prec))
            else:
                print('>epoch=%d, lrate=%.4f, loss=%.3f, val_loss=%.3f' % (epoch, lr, cost, val_cost))
        with open('minmax.json', 'w+') as json_file:  
            json.dump(minmax, json_file)
        if visu:
            plt.plot(plot_tr_loss)
            plt.plot(plot_tr_acc)
            plt.plot(plot_val_loss)
            plt.show()
        return self.network

args = argparse.ArgumentParser("Statistic description of your data file")
args.add_argument("file", help="File to descripte", type=str)
args.add_argument("-e", "--epoch", help="The number of iterations to go through the regression", default=5000)
args.add_argument("-l", "--learning", help="The learning rate to use during the regression", default=0.2, type=float)
args.add_argument("-v", "--visu", help="Visualize functions", action="store_true", default=False)
args.add_argument("-n", "--network", help="Specific network input", type=str, default=False)
args.add_argument("-d", "--details", help="Display more performance metrics", action="store_true", default=False)
args = args.parse_args()

if os.path.isfile(args.file):
    try:
        df = pd.read_csv(args.file, sep=',')
        df = df.dropna()
        df = df.iloc[:, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]]
        network = Train(df, args.epoch, args.learning, args.visu).train(args.network, args.details)
        
    except Exception as e:
        raise(e)
        sys.stderr.write(str(e) + '\n')
        sys.exit()
else:
    sys.stderr.write("Invalid input\n")
    sys.exit(1)