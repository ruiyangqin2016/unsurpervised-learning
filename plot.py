import numpy as np
import matplotlib.pyplot as plt

BP_INFILE = './NN_OUTPUT/BACKPROP_LOG.txt'
GA_INFILE = ['./NN_OUTPUT/GA_50_10_10_LOG_10.txt',
             './NN_OUTPUT/GA_50_10_10_LOG_38.txt',
             './NN_OUTPUT/GA_50_10_20_LOG_10.txt',
             './NN_OUTPUT/GA_50_10_20_LOG_38.txt',
             './NN_OUTPUT/GA_50_20_10_LOG_10.txt',
             './NN_OUTPUT/GA_50_20_10_LOG_38.txt',
             './NN_OUTPUT/GA_50_20_20_LOG_10.txt',
             './NN_OUTPUT/GA_50_20_20_LOG_38.txt']
RHC_INFILE = ['./NN_OUTPUT/RHC_LOG_10.txt','./NN_OUTPUT/RHC_LOG_38.txt']
SA_INFILE = ['./NN_OUTPUT/SA0.7_LOG_10.txt',
             './NN_OUTPUT/SA0.7_LOG_38.txt',
             './NN_OUTPUT/SA0.15_LOG_10.txt',
             './NN_OUTPUT/SA0.15_LOG_38.txt',
             './NN_OUTPUT/SA0.35_LOG_10.txt',
             './NN_OUTPUT/SA0.35_LOG_38.txt',
             './NN_OUTPUT/SA0.55_LOG_10.txt',
             './NN_OUTPUT/SA0.55_LOG_38.txt',
             './NN_OUTPUT/SA0.95_LOG_10.txt',
             './NN_OUTPUT/SA0.95_LOG_38.txt']

FILES=[GA_INFILE[1],GA_INFILE[3],GA_INFILE[5],GA_INFILE[7]]
y_train_plot = []
y_test_plot = []
y_time_plot = []
x_plot = []
legends =['Train Mate=10, Mutate=10', 'Train Mate=10, Mutate=20', 'Train Mate=20, Mutate=10', 'Train Mate=20, Mutate=20']
#legends =['CE=0.7', 'CE=0.15', 'CE=0.35', 'CE=0.55', 'CE=0.95']
color1=['#ff0000','#b0ff00','#00bfff','#bf00ff', '#000000']
color2=['#ff8000','#40ff00','#0040ff','#8f00ff', '#404040']
'''
color1=['#ff0000','#b0ff00','#00bfff','#bf00ff', '#000000']
color2=['#ff8000','#40ff00','#0040ff','#8f00ff', '#404040']
'''
for file in range(len(FILES)):
    trials = 11
    x_axis = []
    y_train_axis = []
    y_test_axis = []
    y_time_axis = []
    trials_end = [0]
    maxdiff = 0
    maxdiff_index = 0
    with open(FILES[file], 'r') as f:
        data = f.readlines()
        start = 1
        for n, line in enumerate(data[start:], 1):
            column=line.rstrip().split(",")
            x_axis.append(float(column[0]))
            y_train_axis.append(float(column[4]))
            y_test_axis.append(float(column[6]))
            if column[-1]=='optimal':
                trials_end.append(n)
                y_time_axis.append(float(column[-2]))
            else:
                y_time_axis.append(float(column[-1]))

        for i in range(len(trials_end)-1):
            diff=trials_end[i+1]-trials_end[i]
            if maxdiff<diff:
                maxdiff=diff
                maxdiff_index=trials_end[i]

        y_train=[]
        y_test=[]
        y_time=[]

        for i in range(trials-1):
            y_train.append(y_train_axis[trials_end[i]:trials_end[i + 1]] + [y_train_axis[trials_end[i + 1] - 1] for j in
                                                                      range(maxdiff - (
                                                                                  trials_end[i + 1] - trials_end[i]))])
            y_test.append(y_test_axis[trials_end[i]:trials_end[i + 1]] + [y_test_axis[trials_end[i + 1] - 1] for j in
                                                                      range(maxdiff - (
                                                                                  trials_end[i + 1] - trials_end[i]))])
            y_time.append(y_time_axis[trials_end[i]:trials_end[i + 1]] + [y_time_axis[trials_end[i + 1] - 1] for j in
                                                                          range(maxdiff - (
                                                                                  trials_end[i + 1] - trials_end[i]))])
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_time = np.array(y_time)
        x=np.array(x_axis[maxdiff_index:maxdiff_index+maxdiff])
        y_train = np.mean(y_train, axis=0)
        y_test = np.mean(y_test, axis=0)
        y_time = np.mean(y_time, axis=0)
        x_plot.append(x)
        y_train_plot.append(y_train)
        y_test_plot.append(y_test)
        y_time_plot.append(y_time)
        plt.plot(x_plot[file], y_train_plot[file], color1[file])
        plt.plot(x_plot[file], y_test_plot[file], color2[file])
        #plt.plot(x_plot[file], y_time_plot[file])
        plt.title('Genetic Algorithm 38 nodes per hidden layer')
        plt.ylabel('Time')
        plt.xlabel('Iterations')
        plt.legend(legends)
plt.show()