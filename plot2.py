import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

FF_INFILE = ['./FLIPFLOP/FLIPFLOP_GA_N_0.2N_0.2N_LOG.txt',
             './FLIPFLOP/FLIPFLOP_GA_N_0.2N_0.5N_LOG.txt',
             './FLIPFLOP/FLIPFLOP_GA_N_0.5N_0.2N_LOG.txt',
             './FLIPFLOP/FLIPFLOP_GA_N_0.5N_0.5N_LOG.txt',
             './FLIPFLOP/FLIPFLOP_MIMIC_N_.5N_0.1_LOG.txt',
             './FLIPFLOP/FLIPFLOP_MIMIC_N_.5N_0.3_LOG.txt',
             './FLIPFLOP/FLIPFLOP_MIMIC_N_.5N_0.5_LOG.txt',
             './FLIPFLOP/FLIPFLOP_MIMIC_N_.5N_0.7_LOG.txt',
             './FLIPFLOP/FLIPFLOP_MIMIC_N_.5N_0.9_LOG.txt',
             './FLIPFLOP/FLIPFLOP_RHC_LOG.txt',
             './FLIPFLOP/FLIPFLOP_SA0.15_LOG.txt',
             './FLIPFLOP/FLIPFLOP_SA0.35_LOG.txt',
             './FLIPFLOP/FLIPFLOP_SA0.55_LOG.txt',
             './FLIPFLOP/FLIPFLOP_SA0.75_LOG.txt',
             './FLIPFLOP/FLIPFLOP_SA0.95_LOG.txt',]
CP_INFILE = ['./CONTPEAKS/CONTPEAKS_GA_N_0.2N_0.2N_LOG.txt',
             './CONTPEAKS/CONTPEAKS_GA_N_0.2N_0.5N_LOG.txt',
             './CONTPEAKS/CONTPEAKS_GA_N_0.5N_0.2N_LOG.txt',
             './CONTPEAKS/CONTPEAKS_GA_N_0.5N_0.5N_LOG.txt',
             './CONTPEAKS/CONTPEAKS_MIMIC_N_.5N_0.1_LOG.txt',
             './CONTPEAKS/CONTPEAKS_MIMIC_N_.5N_0.3_LOG.txt',
             './CONTPEAKS/CONTPEAKS_MIMIC_N_.5N_0.5_LOG.txt',
             './CONTPEAKS/CONTPEAKS_MIMIC_N_.5N_0.7_LOG.txt',
             './CONTPEAKS/CONTPEAKS_MIMIC_N_.5N_0.9_LOG.txt',
             './CONTPEAKS/CONTPEAKS_RHC_LOG.txt',
             './CONTPEAKS/CONTPEAKS_SA0.15_LOG.txt',
             './CONTPEAKS/CONTPEAKS_SA0.35_LOG.txt',
             './CONTPEAKS/CONTPEAKS_SA0.55_LOG.txt',
             './CONTPEAKS/CONTPEAKS_SA0.75_LOG.txt',
             './CONTPEAKS/CONTPEAKS_SA0.95_LOG.txt',]
TSP_INFILE = ['./TSP/TSP_GA_N_0.2N_0.2N_LOG.txt',
             './TSP/TSP_GA_N_0.2N_0.5N_LOG.txt',
             './TSP/TSP_GA_N_0.5N_0.2N_LOG.txt',
             './TSP/TSP_GA_N_0.5N_0.5N_LOG.txt',
             './TSP/TSP_MIMIC_N_.5N_0.1_LOG.txt',
             './TSP/TSP_MIMIC_N_.5N_0.3_LOG.txt',
             './TSP/TSP_MIMIC_N_.5N_0.5_LOG.txt',
             './TSP/TSP_MIMIC_N_.5N_0.7_LOG.txt',
             './TSP/TSP_MIMIC_N_.5N_0.9_LOG.txt',
             './TSP/TSP_RHC_LOG.txt',
             './TSP/TSP_SA0.15_LOG.txt',
             './TSP/TSP_SA0.35_LOG.txt',
             './TSP/TSP_SA0.55_LOG.txt',
             './TSP/TSP_SA0.75_LOG.txt',
             './TSP/TSP_SA0.95_LOG.txt',]

x_label='Points'
y_label='Iteration'
Optimization_Title='Traveling Salesman Optimization'
plot={'Points':0,'Trial':1,'Iteration':2,'Fitness (%)':3,'Time (Seconds)':4}
FILES=[TSP_INFILE[0],TSP_INFILE[4],TSP_INFILE[9],TSP_INFILE[10]]
x_plot = []
y_plot = []
legends =['Genetic Algorithm', 'MIMIC', 'Randomized Hill Climbing', 'Simulated Annealing']
for file in range(len(FILES)):
    trials = 5
    x_axis = []
    y_axis = []
    trials_end = [0]
    maxdiff = 0
    maxdiff_index = 0
    with open(FILES[file], 'r') as f:
        data = f.readlines()
        start = 1
        for n, line in enumerate(data[start:], 1):
            column=line.rstrip().split(",")
            if column[-1]=='optimal':
                x_axis.append(float(column[plot[x_label]]))
                y_axis.append(float(column[plot[y_label]]))

        print(y_axis)
        x_axis=np.array(x_axis)
        y_axis=np.array(y_axis)
        x=[]
        y=[]
        for i in range(int(len(x_axis)/trials)):
            x.append(np.average(x_axis[i*trials:i*trials+trials]))
        for i in range(int(len(y_axis)/trials)):
            y.append(np.average(y_axis[i*trials:i*trials+trials]))

        x_sm = np.array(x)
        y_sm = np.array(y)
        x_smooth = np.linspace(x_sm.min(), x_sm.max())
        y_smooth = spline(x, y, x_smooth)
        x_plot.append(x_smooth)
        y_plot.append(y_smooth)
        plt.plot(x_plot[file], y_plot[file])
        plt.title(Optimization_Title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend(legends)
plt.show()