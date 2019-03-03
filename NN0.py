"""
Backprop NN training on Segmentation data

"""
import os
import csv
import time
import sys
sys.path.append(".\ABAGAIL\ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 24
HIDDEN_LAYER1 = 38
HIDDEN_LAYER2 = 38
HIDDEN_LAYER3 = 38
OUTPUT_LAYER = 5
TRAINING_ITERATIONS = 100001
TRIALS = 11
ERROR_SAMPLE_INTERVAL = 10
HALT_COUNT_MAX = 5000
HALT_COUNT_THRESHOLD = .000001
OUTFILE = './NN_OUTPUT/BACKPROP_LOG.txt'


def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-OUTPUT_LAYER]])
            instance.setLabel(Instance([float(value) for value in row[-OUTPUT_LAYER:]]))
            instances.append(instance)

    return instances


def errorOnDataSet(network, ds, measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getData().argMax()
        predicted = network.getOutputValues().argMax()
        if actual == predicted:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values))
        error += measure.value(output, example)
    MSE = error / float(N)
    acc = correct / float(correct + incorrect)
    return MSE, acc


def train(oa, network, oaName, training_ints, validation_ints, testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print("\nError results for %s\n---------------------------" % (oaName,))
    times = [0]
    halt_count = 0
    lastScore = 0
    iterations = 0
    MSE_trg_last = 0
    MSE_val_last = 0
    MSE_tst_last = 0
    acc_trg_last = 0
    acc_val_last = 0
    acc_tst_last = 0
    for iteration in range(TRAINING_ITERATIONS):
        start = time.clock()
        score = oa.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        if score - lastScore < HALT_COUNT_THRESHOLD:
            halt_count += 1
            if halt_count >= HALT_COUNT_MAX:
                iterations = iteration
                break
        else:
            halt_count = 0
        if score > lastScore:
            lastScore = score
        if iteration % ERROR_SAMPLE_INTERVAL == 0:
            MSE_trg, acc_trg = errorOnDataSet(network, training_ints, measure)
            MSE_val, acc_val = errorOnDataSet(network, validation_ints, measure)
            MSE_tst, acc_tst = errorOnDataSet(network, testing_ints, measure)
            if acc_tst > acc_tst_last:
                MSE_trg_last = MSE_trg
                MSE_val_last = MSE_val
                MSE_tst_last = MSE_tst
                acc_trg_last = acc_trg
                acc_val_last = acc_val
                acc_tst_last = acc_tst
            txt = '{},{},{},{},{},{},{},{}\n'.format(iteration, MSE_trg, MSE_val, MSE_tst, acc_trg, acc_val, acc_tst,
                                                     times[-1]);
            print(txt)
            with open(OUTFILE, 'a+') as f:
                f.write(txt)
    txt = '{},{},{},{},{},{},{},{},{}\n'.format(iterations, MSE_trg_last, MSE_val_last, MSE_tst_last, acc_trg_last, acc_val_last, acc_tst_last,times[-1], 'optimal')
    print(txt)
    with open(OUTFILE, 'a+') as f:
        f.write(txt)


def main():
    """Run this experiment"""
    training_ints = initialize_instances('SeasonsStats_trg.csv')
    testing_ints = initialize_instances('SeasonsStats_test.csv')
    validation_ints = initialize_instances('SeasonsStats_val.csv')
    for t in range(TRIALS):
        factory = BackPropagationNetworkFactory()
        measure = SumOfSquaresError()
        data_set = DataSet(training_ints)
        relu = RELU()
        rule = RPROPUpdateRule()
        oa_names = ["Backprop"]
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1,HIDDEN_LAYER2,HIDDEN_LAYER3, OUTPUT_LAYER],relu)
        train(BatchBackPropagationTrainer(data_set, classification_network, measure, rule), classification_network,'Backprop', training_ints, validation_ints, testing_ints, measure)


if __name__ == "__main__":
    with open(OUTFILE, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg', 'acc_val','acc_tst', 'elapsed'))
    main()
