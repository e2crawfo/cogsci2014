
import pickle
import numpy as np
from mytools import hrr

def analyze_edge_test_similarity(test_fname):
    with open(test_fname, 'rb') as f:
        data = pickle.load(f)

    correct_vectors = data['correct_vectors']
    testing_time = data['testing_time'] * 1000
    num_tests = data['num_tests']

    start_time = 0
    end_time = testing_time
    cushion = 100

    def make_sim_func(h):
        def sim(vec):
            return h.compare(hrr.HRR(data=vec))
        return sim

    means = []

    for i in range(num_tests):
        output = data['output_decoded'][start_time+cushion:end_time, :]
        sim_func = make_sim_func(hrr.HRR(data=correct_vectors[i]))
        sims = np.array([sim_func(vec) for vec in output])

        means.append(np.mean(sims))

        start_time = end_time
        end_time += testing_time

    return means, data

def analyze_edge_test_accuracy(test_fname):
    with open(test_fname, 'rb') as f:
        data = pickle.load(f)

    correct_vectors = data['correct_vectors']
    testing_time = data['testing_time'] * 1000
    num_tests = data['num_tests']

    start_time = 0
    end_time = testing_time
    cushion = 100
    threshold = 0.7

    correct = 0

    for i in range(num_tests):
        output = data['output_decoded'][start_time+cushion:end_time, :]
        correct_hrr = hrr.HRR(data=correct_vectors[i])
        dot = hrr.HRR(data=np.mean(output, axis=0)).compare(correct_hrr)

        if dot > threshold:
            correct += 1

        start_time = end_time
        end_time += testing_time

    return correct, data


