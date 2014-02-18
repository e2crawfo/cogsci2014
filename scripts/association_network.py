import pickle
import nengo
import numpy as np
from mytools import extract_probe_data, fh
from build import build_learning_cleanup, build_cleanup_oja, build_cleanup_pes

import logging
logging.getLogger(__name__)

class Parameters(object):
    def __init__(self, **kwds):
        self.__dict__.update(**kwds)

    def make_filename(title, dir, ext='', usetime=False):
        return fh.make_filename(title, directory, self.__dict__, False)

class LearnableAssociationNetwork(object):
    def __init__(self):
        self.has_learned = False
        self.oja_connection_weights = None
        self.pes_decoders = None
        self.learned_loaded = False
        self.simulator = None

        self.stored_func = lambda x: x
        self.address_func = lambda x: x
        self.model = None
        self.parameters = None

    def set_parameters(self, p):
        self.parameters = p

    def set_vectorized_graph(self, vg):
        self.vectorized_graph = vg

    def build(self):
        if self.parameters is None:
            print "Can't build, parameters not set"
            return

        self._build(**self.parameters.__dict__)

    def _build(self, seed, dim, DperE, NperD, cleanup_n, cleanup_params, ensemble_params,
                  oja_learning_rate, oja_scale, pre_tau, post_tau, pes_learning_rate, rate=False, **kwargs):

        num_ensembles = int(dim / DperE)
        NperE = NperD * DperE

        print "Building..."
        model = nengo.Model("Learn cleanup", seed=seed)

        print self.parameters.__dict__

        if rate:
            neurons = nengo.LIFRate(cleanup_n)
        else:
            neurons = nengo.LIF(cleanup_n)

        def make_func(obj, funcname):
            def g(t):
                return getattr(obj, funcname)(t)
            return g

        # ----- Make Input -----
        self.address_func = lambda x: np.zeros(dim)
        self.stored_func = lambda x: np.zeros(dim)
        self.gate_func = lambda x: [0.0]
        self.learn_func = lambda x: True

        address_input = nengo.Node(output=make_func(self, "address_func"))
        stored_input = nengo.Node(output=make_func(self, "stored_func"))

        # ----- Build neural part -----
        #cleanup = build_training_cleanup(dim, num_vectors, neurons_per_vector, intercept=intercept)
        cleanup = nengo.Ensemble(label='cleanup', neurons=neurons,
                              dimensions=dim, **cleanup_params)

        pre_ensembles, pre_decoders, pre_connections = \
                build_cleanup_oja(model, address_input, cleanup, DperE, NperD, num_ensembles,
                                  ensemble_params, oja_learning_rate, oja_scale,
                                  learn=make_func(self, "learn_func"), rate=rate)

        output_ensembles, error_ensembles = \
                build_cleanup_pes(cleanup, stored_input, DperE, NperD, num_ensembles, pes_learning_rate, rate=rate)

        gate = nengo.Node(output=make_func(self, "gate_func"))
        for ens in error_ensembles:
            nengo.Connection(gate, ens.neurons, transform=-10 * np.ones((NperE, 1)))

        # ----- Build probes -----
        self.address_input_p = nengo.Probe(address_input, 'output')
        self.stored_input_p = nengo.Probe(stored_input, 'output')
        self.pre_probes = [nengo.Probe(ens, 'decoded_output', filter=0.1) for ens in pre_ensembles]
        self.cleanup_s = nengo.Probe(cleanup, 'spikes')
        self.output_probes =  [nengo.Probe(ens, 'decoded_output', filter=0.1) for ens in output_ensembles]

        self.model = model


    def learn(self, sim_length, address_func, stored_func):

        if self.has_learned:
            print "Learning already complete."
            return

        self.address_func = address_func
        self.stored_func = stored_func
        self.gate_func = lambda x: [0.0]
        self.learn_func = lambda x: True

        print "Learning..."

        self.simulator = nengo.Simulator(self.model, dt=0.001)
        self.simulator.run(sim_length)

        self.has_learned = True


    def test(self, sim_length, address_func):
        if not self.has_learned:
            print "Cannot test, hasn't learned yet."
            return None

        if not self.model:
            print "Cannot test, model hasn't been created."
            return None

        if self.simulator is None:
            self.simulator = nengo.Simulator(self.model, dt=0.001)

            input_connections = [conn for conn in self.simulator.model.connections
                                    if conn.pre.label.startswith('pre')
                                    and not isinstance(conn.post, nengo.Probe)]

            for conn in input_connections:
                transform = self.simulator._sigdict[conn.transform_signal]
                transform[...] = self.oja_connection_weights[conn.pre.label]

            output_connections = [conn for conn in self.simulator.model.connections
                                    if conn.pre.label.startswith('cleanup')
                                    and not isinstance(conn.post, nengo.Probe)]

            for conn in output_connections:
                decoder = self.simulator._sigdict[conn.decoder_signal]
                decoder[...] = self.pes_decoders[conn.post.label]
        else:
            self.simulator.reset()

        self.address_func = address_func
        self.stored_func = lambda x: np.zeros(self.parameters.dim)
        self.gate_func = lambda x: [1.0]
        self.learn_func = lambda x: False

        print "Testing..."

        self.simulator.run(sim_length)


    def save_learned_data(self, fname):
        if not self.has_learned:
            print "Cannot save, hasn't learned yet."
            return None

        if not self.simulator:
            print "Cannot save, no simulator."
            return None

        input_connections = [conn for conn in self.simulator.model.connections
                                if conn.pre.label.startswith('pre')
                                and not isinstance(conn.post, nengo.Probe)]
        weights = {conn.pre.label: self.simulator._sigdict[conn.transform_signal]
                      for conn in input_connections}

        output_connections = [conn for conn in self.simulator.model.connections
                                if conn.pre.label.startswith('cleanup')
                                and not isinstance(conn.post, nengo.Probe)]
        decoders = {conn.post.label: self.simulator._sigdict[conn.decoder_signal]
                      for conn in output_connections}

        with open(fname, 'wb') as f:
            pickle.dump((self.parameters, weights, decoders, self.vectorized_graph), f)


    def load_learned_data(self, fname):
        try:
            with open(fname, 'rb') as f:
                ret = pickle.load(f)
                self.parameters = ret[0]
                self.build()
                self.oja_connection_weights = ret[1]
                self.pes_decoders = ret[2]
                self.vectorized_graph = ret[3]
                self.has_learned = True
                self.learning_loaded = True
        except Exception as E:
            print E
            print "Couldn't load."
            return None


    def extract_data(self):

        simulator = self.simulator
        t = self.simulator.trange()

        address_input, _ = extract_probe_data(t, simulator, self.address_input_p)
        stored_input, _ = extract_probe_data(t, simulator, self.stored_input_p)
        pre_decoded, _ = extract_probe_data(t, simulator, self.pre_probes)
        output_decoded, _ = extract_probe_data(t, simulator, self.output_probes)
        cleanup_spikes, _ = extract_probe_data(t, simulator, self.cleanup_s, spikes=True)

        return dict(t=t, address_input=address_input, stored_input=stored_input,
                      pre_decoded=pre_decoded, cleanup_spikes=cleanup_spikes,
                      output_decoded=output_decoded)



