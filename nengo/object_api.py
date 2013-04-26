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

class SelfDependencyError(Exception):
    """Network cannot be simulated because some node input depends on the
    node's own output on the same time-step."""


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


class Var(object):
    def __init__(self, name=None, size=1, dtype=float, shape=None):
        """
        A variable that will contain some numeric data.

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
        clsname = self.__class__.__name__
        if self.name:
            return '%s{%s}' % (clsname, self.name)
        else:
            return '%s{%s}' % (clsname, id(self))

    def __repr__(self):
        return str(self)


class Node(object):
    def __init__(self):
        self.outputs = OrderedDict()
        self.inputs = OrderedDict()

    @property
    def output(self):
        return self.outputs['X']

    def add_to_network(self, network):
        network.nodes.append(self)

class Probe(Node):
    def __init__(self, target):
        Node.__init__(self)
        self.inputs['target'] = target
        self.outputs['stats'] = Var() # -- no default output

    @property
    def target(self):
        return self.inputs['target']

    @property
    def stats(self):
        return self.outputs['stats']

    def add_to_network(self, network):
        network.probes.append(self)


class Filter(Node):
    def __init__(self, var, tau):
        """
        tau: float
        """
        Node.__init__(self)
        self.tau = tau
        self.inputs['var'] = var
        self.outputs['X'] = Var()

    def add_to_network(self, network):
        network.filters.append(self)


class Connection(object):
    def __init__(self, src, dst):
        """
        Parameters
        ----------
        :param Var src:
        :param Var dst: 
        """
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.inputs['X'] = src
        self.outputs['X'] = dst

    def add_to_network(self, network):
        network.connections.append(self)


class Network(object):
    def __init__(self):
        self.probes = []
        self.connections = []
        self.nodes = []
        self.networks = []
        self.filters = []

    def add(self, thing):
        thing.add_to_network(self)
        return thing

    @property
    def all_probes(self):
        if self.networks: raise NotImplementedError()
        return list(self.probes)

    @property
    def all_connections(self):
        if self.networks: raise NotImplementedError()
        return list(self.connections)

    @property
    def all_nodes(self):
        if self.networks: raise NotImplementedError()
        return list(self.nodes)

    @property
    def all_filters(self):
        if self.networks: raise NotImplementedError()
        return list(self.filters)

    @property
    def members(self):
        rval = []
        rval.extend(self.nodes)
        rval.extend(self.connections)
        rval.extend(self.probes)
        rval.extend(self.filters)
        return rval

    @property
    def all_members(self):
        rval = []
        rval.extend(self.all_nodes)
        rval.extend(self.all_connections)
        rval.extend(self.all_probes)
        rval.extend(self.all_filters)
        return rval





class Neurons(Node):
    def __init__(self, size, input_current):
        """
        :param int size:
        :param Var input_current:
        """
        Node.__init__(self)
        self.size = size
        self.inputs['input_current'] = input_current
        self.outputs['X'] = Var()

    @property
    def input_current(self):
        return self.inputs['input_current']

    @input_current.setter
    def input_current(self, val):
        self.inputs['input_current'] = val


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
        if input_current is None:
            input_current = Var()
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

    # TODO: Python magic to support
    # alpha = inputs_getter_setter('alpha')

    @property
    def alpha(self):
        return self.inputs['alpha']

    @alpha.setter
    def alpha(self, val):
        self.inputs['alpha'] = val

    @property
    def j_bias(self):
        return self.inputs['j_bias']

    @j_bias.setter
    def j_bias(self, val):
        self.inputs['j_bias'] = val

    @property
    def voltage(self):
        return self.inputs['voltage']

    @voltage.setter
    def voltage(self, val):
        self.inputs['voltage'] = val

    @property
    def refractory_time(self):
        return self.inputs['refractory_time']

    @refractory_time.setter
    def refractory_time(self, val):
        self.inputs['refractory_time'] = val




class NetworkMember(object):
    def add_to_network(self, network):
        raise NotImplementedError('override in subclass')



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



class LearnedConnection(Connection):
    def __init__(self, src, dst, error_signal):
        Connection.__init__(self, src, dst)
        self.error_signal = error_signal


class hPES_Connection(LearnedConnection):
    theta_tau = 0.02
    unsupervised_rate_factor = 10.
    supervision_ratio = 1.0
    def __init__(self, src, dst, error_signal,
                 theta_tau=theta_tau,
                 unsupervised_rate_factor=unsupervised_rate_factor,
                 supervision_ratio=supervision_ratio,
                ):
        LearnedConnection.__init__(self, src, dst, error_signal)
        self.theta_tau = theta_tau
        self.unsupervised_rate_factor = unsupervised_rate_factor
        self.supervision_ratio = supervision_ratio

        self.seed = 123

        self.gains = Var()
        self.theta = Var()
        self.src_filtered = Var()
        self.dst_filtered = Var()
        self.weight_matrix = Var()
        self.supervised_learning_rate = Var()


simulation_time = Var('time')
simulation_stop_now = Var('stop_when')


class SimulatorBase(object):

    _backends = {}

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

def Simulator(*args, **kwargs):
    backend = kwargs.pop('backend', 'reference')
    if backend not in SimulatorBase._backends:
        if backend == 'numpy':
            import simulator_numpy
        else:
            raise ValueError('backend "%s" not recognized, did you remember to'
                ' import the python module that implements that backend?' %
                backend)
    return SimulatorBase._backends[backend](*args, **kwargs)
