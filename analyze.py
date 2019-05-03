import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

try:
    df = pd.read_csv('data.csv', sep=',')
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(df.dropna(), hue = 'M')
    plt.tight_layout()
    plt.savefig('pair_plot.pdf')
    plt.show()
except Exception as e:
    sys.stderr.write(str(e) + '\n')
    sys.exit()