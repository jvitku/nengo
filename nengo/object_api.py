import math
import random

try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        # -- Fall back on un-ordered dictionaries
        OrderedDict = dict


class Var(object):
    def __init__(self, name=None, size=1, dtype=float, shape=None):
        """
        Parameters
        ----------

        string: identifier, not necessarily unique

        int: number of elements

        string: nature of numbers

        tuple of int: logical shape of array (optional)
        """
        self.name = name
        self.size = size
        self.dtype = dtype
        self.shape = shape

    def __str__(self):
        if self.name:
            return 'Var{%s}' % self.name
        else:
            return 'Var{%s}' % id(self)

    def __repr__(self):
        return str(self)


class Distribution(object):
    @property
    def dist_name(self):
        return self.__class__.__name__.lower()


class Uniform(Distribution):
    def __init__(self, low, high, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.rng = Var()


class Gaussian(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class Neurons(object):
    def __init__(self, size, input_current=None):
        self.size = size
        self.output = Var()        # -- a Var
        self.input_current = input_current # -- a Var


class LIFNeurons(Neurons):
    def __init__(self, size,
            input_current=None,
            tau_rc=0.02,
            tau_ref=0.002,
            max_rate=Uniform(200, 400),
            intercept=Uniform(-1, 1),
            seed=None):
        """
        Parameters
        ----------
        :param int size: number of neurons in this population
        :param float tau_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        """
        Neurons.__init__(self, size, input_current)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.max_rate = max_rate
        self.intercept = intercept
        self.seed = seed
        self.alpha = Var()
        self.j_bias = Var()
        self.voltage = Var()
        self.refractory_time = Var()


class Filter(object):
    def __init__(self, tau):
        """
        tau: float
        """
        self.tau = tau
        self.output = Var()


class NetworkMember(object):
    def add_to_network(self, network):
        raise NotImplementedError('override in subclass')


class Ensemble(NetworkMember):
    DEFAULT_ARRAY_SIZE = 1
    def __init__(self, dimensions, array_size=DEFAULT_ARRAY_SIZE):
        self.dimensions = int(dimensions)
        self.array_size = int(array_size)

        self.origin = OrderedDict()

        self.decoded_input = OrderedDict()

        # set up a dictionary for encoded_input connections
        self.encoded_input = OrderedDict()

    def add_to_network(self, network):
        network.ensembles.append(self)


class DirectEnsemble(Ensemble):
    def __init__(self, dimensions,
            array_size=Ensemble.DEFAULT_ARRAY_SIZE,
            ):
        Ensemble.__init__(dimensions, array_size)
        pass


class NeuronEnsemble(Ensemble):
    def __init__(self,
            neurons,
            dimensions,
            array_size=Ensemble.DEFAULT_ARRAY_SIZE,
            learned_terminations=None

            ):
        """
        learned terminations: list
        """
        Ensemble.__init__(self, dimensions, array_size)
        self.neurons = neurons
        if learned_terminations is None:
            learned_terminations = []
        self.learned_terminations = learned_terminations

    @property
    def spikes(self):
        return self.neurons.output

    @property
    def num_neurons(self):
        # XXX: divide by array_size?
        return self.neurons.size


class Node(object):
    def __init__(self):
        self.outputs = OrderedDict()
        self.inputs = OrderedDict()

    @property
    def output(self):
        return self.outputs['X']

    def add_to_network(self, network):
        network.nodes.append(self)


class TimeNode(Node):
    def __init__(self, func, output=None, name=None):
        Node.__init__(self)
        self.func = func
        if output is None:
            output = Var()
        self.outputs['X'] = output
        self.inputs['time'] = simulation_time
        if output.name is None:
            output.name = name
        self.name = name


class PiecewiseNode(Node):
    def __init__(self, table):
        """
        Parameters
        ----------
        table: Dictionary for piecewise lookup
        """
        self.table = table


class Connection(object):
    def __init__(self, src, dst,
            ):
        self.src = src
        self.dst = dst

    def add_to_network(self, network):
        network.connections.append(self)


class Probe(object):
    def __init__(self, target):
        self.target = target
        self.stats = Var()

    def add_to_network(self, network):
        network.probes.append(self)


class Network(object):
    def __init__(self):
        self.probes = []
        self.ensembles = []
        self.connections = []
        self.nodes = []
        self.networks = []

    def add(self, thing):
        thing.add_to_network(self)
        return thing

    @property
    def all_probes(self):
        if self.networks: raise NotImplementedError()
        return list(self.probes)

    @property
    def all_ensembles(self):
        if self.networks: raise NotImplementedError()
        return list(self.ensembles)

    @property
    def all_connections(self):
        if self.networks: raise NotImplementedError()
        return list(self.connections)

    @property
    def all_nodes(self):
        if self.networks: raise NotImplementedError()
        return list(self.nodes)

    @property
    def members(self):
        rval = []
        rval.extend(self.nodes)
        rval.extend(self.connections)
        rval.extend(self.ensembles)
        rval.extend(self.probes)
        return rval

    @property
    def all_members(self):
        rval = []
        rval.extend(self.all_nodes)
        rval.extend(self.all_connections)
        rval.extend(self.all_ensembles)
        rval.extend(self.all_probes)
        return rval


simulation_time = Var('time')
simulation_stop_now = Var('stop_when')


class Simulator(object):
    def __init__(self, network):
        self.network = network
        self.simulation_time = 0.0

    def reset(self):
        self.simulation_time = 0.0

    def run_steps(self, steps, dt):
        """
        Returns a dictionary mapping targets that have been probed
        to a list of either lists or arrays.
        """
        raise NotImplementedError('Use a simulator subclass')


