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
    def __init__(self, data, iterations, lr, visu):
        # data['t0'] = np.ones(data.loc[:, 'Hogwarts House'].shape[0])
        self.data = data
        self.lr = lr
        self.visu = visu
        self.iterations = iterations
        self.predictions = {}
        self.costs = {}

    def train(self):
        sns.set(style="ticks", color_codes=True)
        sns.pairplot(self.data, hue = 'M')
        plt.tight_layout()
        plt.savefig('pair_plot_selected.pdf')





args = argparse.ArgumentParser("Statistic description of your data file")
args.add_argument("file", help="File to descripte", type=str)
args.add_argument("-i", "--iter", help="The number of iterations to go through the regression", default=10000, type=int)
args.add_argument("-l", "--learning", help="The learning rate to use during the regression", default=0.01, type=float)
args.add_argument("-v", "--visu", help="Visualize functions", action="store_true", default=False)
args = args.parse_args()

if os.path.isfile(args.file):
    try:
        df = pd.read_csv(args.file, sep=',')
        df = df.dropna()
        df = df.iloc[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]]
        Train(df, args.iter, args.learning, args.visu).train()
        
    except Exception as e:
        raise(e)
        sys.stderr.write(str(e) + '\n')
        sys.exit()
else:
    sys.stderr.write("Invalid input\n")
    sys.exit(1)