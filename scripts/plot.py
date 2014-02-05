import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mytools import hrr, nengo_plot_helper, apply_funcs
import analyze

def edge_accuracy_plot(filenames, plot_fname, show=False):

    acc = []
    indices = []
    for fn in filenames:
        correct, data = analyze.analyze_edge_test_accuracy(fn)
        acc.append(np.true_divide(correct, data['num_tests']))
        indices.append(data['num_vectors'])

    plt.plot(np.array(indices), np.array(acc))

    plt.savefig(plot_fname)

    if show:
        plt.show()

def edge_similarity_plot(filenames, plot_fname, show=False):
    means = []
    indices = []
    for fn in filenames:
        mean_sims, data = analyze.analyze_edge_test_similarity(fn)
        means.append(np.mean(mean_sims))
        indices.append(data['num_vectors'])

    plt.plot(np.array(indices), np.array(means))

def path_accuracy_plot(filenames):
    pass

def path_similarity_plot(filenames):
    pass

def load_simulation_data(fname, data):
    with open(fname, 'rb') as f:
        filedata = pickle.load(f)

    if len(data) == 0:
        data.update(filedata)
    else:
        for key in filedata:
            if key == 't':
                filedata[key] += data[key][-1] + data[key][1]

            if key in data:
                data[key] = np.concatenate((data[key], filedata[key]), axis=0)
            else:
                data[key] = filedata[key]


def simulation_plot(plot_fname, learning_fname='', testing_fname='', show=False):
    mpl.rcParams['lines.linewidth'] = 1.0

    if not (learning_fname or testing_fname):
        print "Couldn't make plot, no filenames given"
        return

    data = {}

    if learning_fname:
        load_simulation_data(learning_fname, data)

    if testing_fname:
        load_simulation_data(testing_fname, data)

    num_plots = 2
    fig = plt.figure(figsize=(6, 2.5))

    show_sims = 'vg' in data
    if show_sims:
        num_plots += 2

        vg = data['vg']
        def make_sim_func(h):
            def sim(vec):
                return h.compare(hrr.HRR(data=vec))
            return sim

        address_sim_funcs = [make_sim_func(vg.id_vectors[n]) for n in vg.G]
        stored_sim_funcs = [make_sim_func(vg.hrr_vectors[n]) for n in vg.G]

    offset = num_plots * 100 + 10 + 1
    plt.subplots_adjust(right=0.93, top=0.98, bottom=0.11, left=0.07, hspace=0.01)

    t = data['t']

    if show_sims:
        address_sims = apply_funcs(address_sim_funcs, data['address_input'])
        ax, offset = nengo_plot_helper(offset, t, address_sims, yticks=[-1,0,1])
        plt.ylim((-1.1, 1.1))
        plt.axhline(y = 1.0, color='black', linestyle='--', zorder=0)

        stored_sims = apply_funcs(stored_sim_funcs, data['output_decoded'])
        ax, offset = nengo_plot_helper(offset, t, stored_sims, yticks=[-1,0,1], removex=True)
        plt.ylim((-1.1, 1.1))
        plt.axhline(y = 1.0, color='black', linestyle='--', zorder=0)

    #ax, offset = nengo_plot_helper(offset, t, data['address_input'], yticks=[])
    #ax, offset = nengo_plot_helper(offset, t, data['stored_input'], yticks=[])
    #ax, offset = nengo_plot_helper(offset, t, data['pre_decoded'], yticks=[])
    testing_time = data['testing_time'] * data['num_tests'] * 1000
    cleanup_spikes = spike_sorter(data['cleanup_spikes'], data['num_vectors'] + 1,
                        slice=np.index_exp[:-testing_time,:], binsize=20)
    ax, offset = nengo_plot_helper(offset, t, cleanup_spikes, spikes=True, yticks=[])
    ax, offset = nengo_plot_helper(offset, t, data['output_decoded'], yticks=[], removex=False)

    plt.savefig(plot_fname)

    if show:
        plt.show()

