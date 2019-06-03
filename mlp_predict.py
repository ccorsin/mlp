import json
import argparse
import sys
import os
import csv
import pandas as pd
import numpy as np

def normalize_dataset(dataset, minmax):
    normalized_data = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    return normalized_data

def build_excpected(data):
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.column_stack((e_x.sum(axis=1),e_x.sum(axis=1)))

def relu(x):
    return abs(x) * (x > 0)

def forward_propagation(data, network):
    inputs = data
    caches = []
    i = 0
    for layer in network:
        Z = np.dot(inputs, layer['W'].T) + layer['b'].T
        layer['Z'] = Z
        if i < len(network) - 1:
            layer['A'] = relu(Z)
        else:
            layer['A'] = softmax(Z)
        layer['A_prev'] = inputs
        inputs = layer['A']
        cache = (layer['A'], layer['W'], layer['b'], layer['A_prev'], layer['Z'])
        caches.append(cache)
        i += 1
    return inputs, caches

def predict(inputs, network):
    outputs = forward_propagation(inputs, network)
    return outputs

def evaluate_perfo(prediction, y):
    error = np.sum((np.argmax(prediction, axis=1) - np.argmax(y, axis=1)) ** 2)
    acc = (len(y) - error) * 100 / len(y)
    print (error, len(y))
    return acc

if __name__ == '__main__':
    args = argparse.ArgumentParser("Predict cancer from data")
    args.add_argument("data_set", help="File to descripte", type=str)
    args.add_argument("minmax", help="Normalization parameters", type=str)
    args.add_argument("model", help="Trained parameters", type=str)
    args = args.parse_args()
    if os.path.isfile(args.data_set):
        try:
            df = pd.read_csv(args.data_set, sep=',')
            if os.path.isfile(args.minmax):
                try:
                    json_file = open(args.minmax)
                    data_json = json.load(json_file)
                    minmax = data_json
                    if os.path.isfile(args.model):
                        try:
                            network = np.load(args.model)
                        except Exception as e:
                            sys.stderr.write("Le fichier n'est pas correct1\n")
                            sys.exit(1)
                    else:
                        sys.stderr.write("Le fichier n'est pas correct2\n")
                        sys.exit(1)
                except Exception as e:
                    sys.stderr.write("Le fichier n'est pas correct3\n")
                    sys.exit(1)
            else:
                sys.stderr.write("Le fichier n'est pas correct4\n")
                sys.exit(1)
        except Exception as e:
            sys.stderr.write("Le fichier n'est pas correct5\n")
            sys.exit(1)
    else:
        sys.stderr.write("Le fichier n'est pas correct6\n")
        sys.exit(1)
    df = df.dropna()
    df = df.iloc[:, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]]
    y_test = build_excpected(df).values
    data_test = df.iloc[:, 1:]
    std_data_test = normalize_dataset(data_test, minmax)
    prediction, _ = predict(std_data_test, network)
    accuracy = evaluate_perfo(prediction, y_test)
    print (accuracy)
    