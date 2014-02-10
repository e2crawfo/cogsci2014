import networkx as nx
import matplotlib.pyplot as plt
from mytools import hrr, nf
import random

def semantic_network(n, seed=1):
    #G = nx.scale_free_graph(n, alpha=0.05, gamma=0.41, delta_out=0.2, delta_in=0, seed=seed)
    G = nx.scale_free_graph(n,seed=seed)
    G.remove_edges_from(G.selfloop_edges())
    G = nx.DiGraph(G)

    return G

def simple_network(n, seed=1):
    G = nx.cycle_graph(n, nx.DiGraph())
    return G

def draw_semantic_network(G, edge_labels=None):
    #pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

def make_id_vectors(G, D):
    vocab = hrr.Vocabulary(D)
    id_vectors = {n: vocab.parse('node'+str(n)) for n in G.nodes_iter()}
    return id_vectors

def make_edge_vectors(G, D):
    vocab = hrr.Vocabulary(D)
    max_out_degree = max(G.out_degree(G.nodes_iter()).values())
    edge_vectors = {i: vocab.parse('edge'+str(i)) for i in range(max_out_degree)}
    return edge_vectors

def make_hrr_vectors(G, id_vectors, edge_vectors):
    max_out_degree = len(edge_vectors)

    index_dict = {}
    for n in G.nodes_iter():
        edges = G.edges(n)
        indices = random.sample(range(max_out_degree), len(edges))
        index_dict.update(dict(zip(edges, indices)))

    nx.set_edge_attributes(G, 'index', index_dict)

    hrrvecs = {}
    for n in G.nodes_iter():
        edges = G.edges(n)
        if not edges:
            hrr_vec = hrr.HRR(len(id_vectors.values()[0]))
        else:
            components = [id_vectors[e[1]].convolve(edge_vectors[index_dict[e]])
                          for e
                          in edges]
            hrrvec = sum(components[1:], components[0])
            hrrvec.normalize()

        hrrvecs[n] = hrrvec

    return hrrvecs

def set_edge_weights(G, hrr_vectors, id_vectors, edge_vectors):

    index_dict = nx.get_edge_attributes(G, 'index')
    weight_dict = {}

    for n in G.nodes_iter():
        edges = G.edges(n)

        target_vectors = [id_vectors[e[1]] for e in edges]

        hrr_vec = hrr_vectors[n]
        trace_vectors = [hrr_vec.convolve(~edge_vectors[index_dict[e]]) for e in edges]

        weights = [round(target.compare(trace), 4)
                   for target, trace
                   in zip(target_vectors, trace_vectors)]
        weight_dict.update(dict(zip(edges, weights)))

    nx.set_edge_attributes(G, 'weight', weight_dict)
    return weight_dict

class VectorizedGraph(object):
    def __init__(self, D, N, seed=1, simple=False, draw=True, save=True):
        random.seed(seed)

        if simple:
            G = simple_network(N, seed=seed)
        else:
            G = semantic_network(N, seed=seed)

        id_vectors = make_id_vectors(G, D)

        edge_vectors = make_edge_vectors(G, D)

        hrr_vectors = make_hrr_vectors(G, id_vectors, edge_vectors)
        weight_dict = set_edge_weights(G, hrr_vectors, id_vectors, edge_vectors)

        if draw:
            plt.figure()
            draw_semantic_network(G, edge_labels=weight_dict)
            if save:
                plt.savefig("graphs/sn_D_%g_N_%g_s_seed_%g.pdf" % (D, N, seed))
            plt.show()

        self.num_vectors = N
        self.hrr_vectors = hrr_vectors
        self.id_vectors = id_vectors
        self.edge_vectors = edge_vectors
        self.G = G

    def training_schedule(self, training_time):
        address_gens = [nf.output(100, True, self.id_vectors[n].v, False)
                        for n in self.G]
        stored_gens = [nf.output(100, True, self.hrr_vectors[n].v, False)
                        for n in self.G]

        address_times = [training_time] * self.num_vectors
        stored_times = [training_time] * self.num_vectors

        address_func = nf.make_f(address_gens, address_times)
        stored_func = nf.make_f(stored_gens, stored_times)

        sim_time = sum(address_times)

        return (sim_time, address_func, stored_func)

    def edge_testing_schedule(self, testing_time, num_tests, node_order=None):
        if node_order is not None:
            nodes = list(self.G)
            nodes = [nodes[i] for i in node_order]
            edges = [random.sample(list(self.G.edges_iter(n, data=True)), 1)[0] for n in nodes]
        else:
            edges = list(self.G.edges_iter(data=True))
            edges = [edges[int(random.random() * len(edges))] for i in xrange(num_tests)]

        correct_vectors = [self.hrr_vectors[v].v for u,v,d in edges]
        input_vectors = [self.id_vectors[v].v for u,v,d in edges]

        testing_vectors = [self.hrr_vectors[u].convolve(~self.edge_vectors[d['index']])
                           for u,v,d in edges]
        testing_vectors = map(lambda x: x.v, testing_vectors)

        testing_gens = [nf.output(100, True, tv, False) for tv in testing_vectors]
        testing_times = [testing_time] * num_tests
        testing_func = nf.make_f(testing_gens, testing_times)

        sim_time = sum(testing_times)

        return (sim_time, testing_func, correct_vectors, input_vectors)


    def path_testing_schedule(self, testing_time, num_tests, path_length):
        edges = random.sample(list(self.G.edges_iter(data=True)), num_tests)

        correct_vectors = [hrr_vectors[v] for u,v,d in edges]

        testing_vectors = [hrr_vectors[u].convolve(~edge_vectors[d['index']]) for u,v,d in edges]
        testing_vectors = map(lambda x: x.v, testing_vectors)

        return (sim_time, address_func, correct_vectors)

if __name__ == '__main__':
    seed = 100
    D = 512
    N = 30

    V = VectorizedGraph(D, N, seed, draw=True, save=False)

