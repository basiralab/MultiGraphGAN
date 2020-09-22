import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

# read csv file
def readcsv(filename):
    data = pd.read_csv(filename) 
    c = []
    data = np.array(data)
    for i in range(0,data.shape[0]):
        a = data[i][0]
        b = np.array(list(a.split(" ")))
        c.append(b) 

    return(np.array(c))

# Plot the connectomes
def show_mtrx(m):
    fig, ax = plt.subplots(figsize = (20, 10))

    min_val = round((m.min()), 6)
    max_val = round((m.max()), 6)
    
    cax = ax.matshow(m, cmap=plt.cm.Spectral)
    cbar = fig.colorbar(cax, ticks=[min_val, float((min_val + max_val)/2), max_val])
    cbar.ax.set_yticklabels(['< %.2f'%(min_val), '%.2f'%(float((min_val + max_val)/2)), '> %.2f'%(max_val)])
    plt.title(label="Source graph")
    plt.show()
    

# put it back into a 2D symmetric array
def to_2d(vector):
    size = 35
    x = np.zeros((size,size))
    c = 0
    for i in range(1,size):
        for j in range(0,i):
            x[i][j] = vector[c]
            x[j][i] = vector[c]
            c = c + 1
    return x

# Display the source matrix of the first subject
pred = readcsv("source_graphs.csv")
SG = to_2d(pred[0])
show_mtrx(SG)

# Display the target graph in the domain 1 of the first subject
pred = readcsv("predicted_graphs_1.csv")
TG1 = to_2d(pred[0])
show_mtrx(TG1)

# Display the target graph in the domain 2 of the first subject
pred = readcsv("predicted_graphs_2.csv")
TG2 = to_2d(pred[0])
show_mtrx(TG2)

    