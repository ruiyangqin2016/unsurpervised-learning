import sys
import os
import time

sys.path.append(".\ABAGAIL\ABAGAIL.jar")
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
from shared import SumOfSquaresError, DataSet, Instance
from time import clock
from itertools import product

"""
Commandline parameter(s):
   none
"""

N_min = 20
N_max = 200
maxIters = 100001
increments = 10
numTrials = 5
halt_max = 100
outfile = './CONTPEAKS/CONTPEAKS_@ALG@_LOG.txt'


#############################################################
# RHC
#############################################################
fname = outfile.replace('@ALG@', 'RHC')
with open(fname, 'w') as f:
    f.write('N,T,trial,iterations,fitness,time,fevals\n')
for N in range(N_min, N_max + 1, N_min):
    fill = [2] * N
    ranges = array('i', fill)
    T = 4 * N / 10
    for t in range(numTrials):
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, increments)
        times = [0]
        lastScore = 0
        halt_count = 0
        total_iter = 0
        for i in range(0, maxIters, increments):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            fevals = ef.fevals
            score = ef.value(rhc.getOptimal())
            ef.fevals -= 1
            st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, i, score, times[-1], fevals, halt_count)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)
            if score - lastScore <= 0:
                halt_count += 1
                if halt_count >= halt_max:
                    total_iter = i
                    break
            else:
                halt_count = 0
            if score > lastScore:
                lastScore = score
        optimal = N + (N - T - 1)
        st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, total_iter, lastScore, times[-1], fevals, 'optimal')
        print(st)
        with open(fname, 'a') as f:
            f.write(st)

#############################################################
# SA
#############################################################
# Iterate over cooling exponent
for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
    fname = outfile.replace('@ALG@', 'SA{}'.format(CE))
    with open(fname, 'w') as f:
        f.write('N,T,trial,iterations,fitness,time,fevals\n')
    # Iterate over bitstring length (model complexity)
    for N in range(N_min, N_max + 1, N_min):
        fill = [2] * N
        ranges = array('i', fill)
        T = 4 * N / 10
        # Iterate over number of trials
        for t in range(numTrials):
            # Setup
            ef = ContinuousPeaksEvaluationFunction(T)
            odd = DiscreteUniformDistribution(ranges)
            nf = DiscreteChangeOneNeighbor(ranges)
            hcp = GenericHillClimbingProblem(ef, odd, nf)
            sa = SimulatedAnnealing(1E10, CE, hcp)
            fit = FixedIterationTrainer(sa, increments)
            times = [0]
            lastScore = 0
            halt_count = 0
            total_iter = 0
            # Evaluate at iteration increments
            for i in range(0, maxIters, increments):
                start = clock()
                fit.train()
                elapsed = time.clock() - start
                times.append(times[-1] + elapsed)
                fevals = ef.fevals
                score = ef.value(sa.getOptimal())
                ef.fevals -= 1
                st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, i, score, times[-1], fevals, halt_count)
                print(st)
                with open(fname, 'a') as f:
                    f.write(st)
                # Check for convergence
                if score - lastScore <= 0:
                    halt_count += 1
                    if halt_count >= halt_max:
                        total_iter = i
                        break
                else:
                    halt_count = 0
                if score > lastScore:
                    lastScore = score
            # Optimal value found in trial
            optimal = N + (N - T - 1)
            st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, total_iter, lastScore, times[-1], fevals, 'optimal')
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

#############################################################
# GA
#############################################################
for mate, mutate in product([.5, .2], [.5, .2]):
    fname = outfile.replace('@ALG@', 'GA_N_{}N_{}N'.format(mate, mutate))
    with open(fname, 'w') as f:
        f.write('N,T,trial,iterations,fitness,time,fevals\n')
    for N in range(N_min, N_max + 1, N_min):
        pop = N
        mateN = int(mate * N)
        mutateN = int(mutate * N)
        fill = [2] * N
        ranges = array('i', fill)
        T = 4 * N / 10
        for t in range(numTrials):
            ef = ContinuousPeaksEvaluationFunction(T)
            odd = DiscreteUniformDistribution(ranges)
            nf = DiscreteChangeOneNeighbor(ranges)
            mf = DiscreteChangeOneMutation(ranges)
            cf = SingleCrossOver()
            gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
            ga = StandardGeneticAlgorithm(pop, mateN, mutateN, gap)
            fit = FixedIterationTrainer(ga, increments)
            times = [0]
            lastScore = 0
            halt_count = 0
            total_iter = 0
            for i in range(0, maxIters, increments):
                start = clock()
                fit.train()
                elapsed = time.clock() - start
                times.append(times[-1] + elapsed)
                fevals = ef.fevals
                score = ef.value(ga.getOptimal())
                ef.fevals -= 1
                st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, i, score, times[-1], fevals, halt_count)
                print(st)
                with open(fname, 'a') as f:
                    f.write(st)
                if score - lastScore <= 0:
                    halt_count += 1
                    if halt_count >= halt_max:
                        total_iter = i
                        break
                else:
                    halt_count = 0
                if score > lastScore:
                    lastScore = score
            optimal = N + (N - T - 1)
            st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, total_iter, lastScore, times[-1], fevals, 'optimal')
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

#############################################################
# MIMIC
#############################################################
for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
    fname = outfile.replace('@ALG@', 'MIMIC_N_.5N_{}'.format(m))
    with open(fname, 'w') as f:
        f.write('N,T,trial,iterations,fitness,time,fevals\n')
    for N in range(N_min, N_max + 1, N_min):
        samples = N
        keep = N / 2
        fill = [2] * N
        ranges = array('i', fill)
        T = 4 * N / 10
        for t in range(numTrials):
            ef = ContinuousPeaksEvaluationFunction(T)
            odd = DiscreteUniformDistribution(ranges)
            nf = DiscreteChangeOneNeighbor(ranges)
            mf = DiscreteChangeOneMutation(ranges)
            cf = SingleCrossOver()
            gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
            df = DiscreteDependencyTree(m, ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            mimic = MIMIC(samples, keep, pop)
            fit = FixedIterationTrainer(mimic, increments)
            times = [0]
            lastScore = 0
            halt_count = 0
            total_iter = 0
            for i in range(0, maxIters, increments):
                start = clock()
                fit.train()
                elapsed = time.clock() - start
                times.append(times[-1] + elapsed)
                fevals = ef.fevals
                score = ef.value(mimic.getOptimal())
                ef.fevals -= 1
                st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, i, score, times[-1], fevals, halt_count)
                print(st)
                with open(fname, 'a') as f:
                    f.write(st)
                if score - lastScore <= 0:
                    halt_count += 1
                    if halt_count >= halt_max:
                        total_iter = i
                        break
                else:
                    halt_count = 0
                if score > lastScore:
                    lastScore = score
            optimal = N + (N - T - 1)
            st = '{},{},{},{},{},{},{},{}\n'.format(N, T, t, total_iter, lastScore, times[-1], fevals, 'optimal')
            print(st)
            with open(fname, 'a') as f:
                f.write(st)
