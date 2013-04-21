try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict

import numpy as np

from . import neuron
from . import ensemble_origin
from . import origin
from . import cache
from . import filter
from .simulator import SymbolicSignal
from .simulator import Component
from .hPES_termination import hPESTermination

_ARRAY_SIZE = 1

class Uniform(object):
    def __init__(self, low, high):
        self.type = 'uniform'
        self.low = low
        self.high = high


class Gaussian(object):
    def __init__(self, low, high):
        self.type = 'gaussian'
        self.low = low
        self.high = high


class Base(Component):
    def __init__(self, dimensions, array_size=_ARRAY_SIZE):
        self.dimensions = int(dimensions)
        self.array_size = int(array_size)

        self.origin = OrderedDict()
        self.decoded_input = OrderedDict()
        self.encoded_input = OrderedDict()
        self.learned_terminations = []


class DirectEnsemble(Base):


    def add_origin(self, name, func, **kwargs):
        """
        Parameters:
        name: string
            name will be installed to returned SymbolicSignal

        Returns:
        SymbolicSymbol handle to computed origin
        """
        if func is not None:
            # -- make sure there is an initial_value by calling func(0)
            if 'initial_value' not in kwargs.keys():
                # [func(np.zeros(self.dimensions)) for i in range(self.array_size)]
                init = func(np.zeros(self.dimensions))
                init = np.array([init for i in range(self.array_size)])
                kwargs['initial_value'] = init.flatten()

        if 'dt' in kwargs.keys():
            del kwargs['dt']

        rval = SymbolicSignal(name)
        self.origin[rval] = origin.Origin(func=func, **kwargs) 
        return rval


    def add_encoded_termination(self, name, term, pstc):
        rval = SymbolicSignal(name)
        filt = filter.Filter(pstc, 
            source=encoded_input, 
            shape=(self.array_size, self.neurons))
        self.encoded_input[rval] = filt.state

    def add_decoded_termination(self, name, term, pstc):
        rval = SymbolicSignal(name)
        filt = filter.Filter(pstc, 
            source=decoded_input, 
            shape=(self.array_size, self.dimensions))
        self.decoded_input[rval] = filt.state
        return rval

    def reset(self, state):
        pass

    @property
    def step_inputs(self):
        return self.decoded_input.values()

    @property
    def step_outputs(self):
        return self.origin.values()

    def step(self, state, dt):

        # set up matrix to store accumulated decoded input
        X = np.zeros((self.array_size, self.dimensions))

        # updates is an ordered dictionary of theano variables to update
        for di in self.decoded_input.values(): 
            # add its values to the total decoded input
            X += state[di]

        # if we're calculating a function on the decoded input
        for o in self.origin.values(): 
            if o.func is not None:  
                val = np.float32([o.func(X[i]) for i in range(len(X))])
                state[o.decoded_output] = val.flatten()


class SpikingEnsemble(Base):
    """An ensemble is a collection of neurons representing a vector space.
    """
    
    def __init__(self, neurons, dimensions, array_size=_ARRAY_SIZE,
            neuron_model=None,
            max_rate=(200, 300), intercept=(-1.0, 1.0),
            radius=1.0,
            encoders=None,
            seed=None,
            decoder_noise=0.1,
            noise=None,
            ):
        """Construct an ensemble composed of the specific neuron model,
        with the specified neural parameters.

        :param tuple max_rate:
            lower and upper bounds on randomly generated
            firing rates for each neuron
        :param tuple intercept:
            lower and upper bounds on randomly generated
            x offsets for each neuron
        :param float radius:
            the range of input values (-radius:radius)
            per dimension this population is sensitive to
        :param list encoders: set of possible preferred directions
        :param int seed: seed value for random number generator
        :param string neuron_type:
            type of neuron model to use, options = {'lif'}
        :param int array_size: number of sub-populations for network arrays
        :param float decoder_noise: amount of noise to assume when computing 
            decoder    
        :param string noise_type:
            the type of noise added to the input current.
            Possible options = {'uniform', 'gaussian'}.
            Default is 'uniform' to match the Nengo implementation.
        :param noise: distribution e.g. Uniform
            noise parameter for noise added to input current,
            sampled at every timestep.

        """
        Base.__init__(self, dimensions, array_size)
        self.neurons = int(neurons)
        if seed is None:
            seed = np.random.randint(1000)
        if neurons % array_size:
            raise ValueError('array_size must divide population size',
                    (neurons, array_size))
        self.seed = seed
        self.radius = radius
        self.noise = noise
        self.decoder_noise = decoder_noise
        if neuron_model is None:
            self.neuron_model = neuron.lif.LIFNeuron()
        else:
            self.neuron_model = neuron_model
        self.input_current = SymbolicSignal()
        # XXX is it really worth distinguishing neuron_model from population?
        self.population = self.neuron_model.Population(
                self.array_size * self.neurons,
                input_current=self.input_current)
        self.s_rng = SymbolicSignal()
        self.bias = SymbolicSignal()
        self.alpha = SymbolicSignal()
        self.encoders = SymbolicSignal()

    def reset(self, state):

        # make sure intercept is the right shape
        if isinstance(intercept, (int,float)):
            intercept = [intercept, 1]
        elif len(intercept) == 1:
            intercept.append(1) 

        # compute alpha and bias
        # XXX (specific to neuron model?)
        rng = state[self.s_rng] = np.random.RandomState(seed=self.seed)
        max_rate = self.max_rate
        max_rates = rng.uniform(
            size=(self.array_size, self.neurons),
            low=max_rate[0], high=max_rate[1])  
        threshold = rng.uniform(
            size=(self.array_size, self.neurons),
            low=intercept[0], high=intercept[1])
        alpha, bias = self.neuron_model.make_alpha_bias(max_rates, threshold)

        state[self.bias] = bias.astype('float32')
        state[self.alpha] = alpha.astype('float32')

        # compute encoders
        encoders = self.make_encoders(encoders=encoders, rng=rng)
        # combine encoders and gain for simplification
        state[self.encoders] = (self.encoders.T * alpha.T).T

    def add_encoded_termination(self, name, term, pstc):
        self.encoded_input[name] = filter.Filter(
            pstc, 
            source=encoded_input, 
            shape=(self.array_size, self.neurons))

    def add_decoded_termination(self, name, term, pstc):
        self.decoded_input[name] = filter.Filter(
            pstc, 
            source=decoded_input / self.radius, 
            shape=(self.array_size, self.dimensions))

    def add_learned_termination(self, name, pre, error, pstc, 
                                learned_termination_class=hPESTermination,
                                **kwargs):
        """Adds a learned termination to the ensemble.

        Input added to encoded_input, and a learned_termination object
        is created to keep track of the pre and post
        (self) spike times, and adjust the weight matrix according
        to the specified learning rule.

        :param Ensemble pre: the pre-synaptic population
        :param Ensemble error: the Origin that provides the error signal
        :param float pstc:
        :param learned_termination_class:
        """
        #TODO: is there ever a case we wouldn't want this?
        assert error.dimensions == self.dimensions * self.array_size

        # generate an initial weight matrix if none provided,
        # random numbers between -.001 and .001
        if 'weight_matrix' not in kwargs.keys():
            weight_matrix = np.random.uniform(
                size=(self.array_size * pre.array_size,
                      self.neurons, pre.neurons),
                low=-.001, high=.001)
            kwargs['weight_matrix'] = weight_matrix
        else:
            # make sure it's an np.array
            #TODO: error checking to make sure it's the right size
            kwargs['weight_matrix'] = np.array(kwargs['weight_matrix']) 

        learned_term = learned_termination_class(
            pre=pre, post=self, error=error, **kwargs)

        learn_projections = [np.dot(
            pre.population.output[learned_term.pre_index(i)],  
            learned_term.weight_matrix[i % self.array_size]) 
            for i in range(self.array_size * pre.array_size)]

        # now want to sum all the output to each of the post ensembles 
        # going to reshape and sum along the 0 axis
        learn_output = np.sum( 
            np.reshape(learn_projections, 
            (pre.array_size, self.array_size, self.neurons)), axis=0)
        # reshape to make it (array_size x self.neurons)
        learn_output = np.reshape(learn_output, 
            (self.array_size, self.neurons))

        # the input_current from this connection during simulation
        self.add_termination(name=name, pstc=pstc, encoded_input=learn_output)
        self.learned_terminations.append(learned_term)
        return learned_term

    def add_origin(self, name, func, **kwargs):
        """Create a new origin to perform a given function
        on the represented signal.

        :param string name: name of origin
        :param function func:
            desired transformation to perform over represented signal
        :param list eval_points:
            specific set of points to optimize decoders over for this origin
        """

        if 'eval_points' not in kwargs.keys():
            kwargs['eval_points'] = self.eval_points
        self.origin[name] = ensemble_origin.EnsembleOrigin(
            ensemble=self, func=func, **kwargs)


    def make_encoders(self, encoders=None, rng=None):
        """Generates a set of encoders.

        :param int neurons: number of neurons 
        :param int dimensions: number of dimensions
        :param theano.tensor.shared_randomstreams snrg:
            theano random number generator function
        :param list encoders:
            set of possible preferred directions of neurons

        """
        if encoders is None:
            # if no encoders specified, generate randomly
            encoders = rng.normal(
                size=(self.array_size, self.neurons, self.dimensions))
            assert encoders.ndim == 3
        else:
            # if encoders were specified, cast list as array
            encoders = np.array(encoders).T
            # repeat array until 'encoders' is the same length
            # as number of neurons in population
            encoders = np.tile(encoders,
                (self.neurons / len(encoders) + 1)
                               ).T[:self.neurons, :self.dimensions]
            encoders = np.tile(encoders, (self.array_size, 1, 1))

            assert encoders.ndim == 3
        # normalize encoders across represented dimensions 
        print encoders.shape
        norm = np.sum(encoders * encoders, axis=2)[:, :, None]
        encoders = encoders / np.sqrt(norm)        

        return encoders

    @property
    def step_inputs(self):
        return (self.encoded_input.values()
            + self.decoded_input.values()
            + [self.bias, self.encoders, self.s_rng]
            )

    @property
    def step_outputs(self):
        return [self.input_current]

    def step(self, state, dt):
        """Compute the set of theano updates needed for this ensemble.

        Returns a dictionary with new neuron state,
        termination, and origin values.

        :param float dt: the timestep of the update
        """
        
        ### find the total input current to this population of neurons

        # set up matrix to store accumulated decoded input
        X = np.zeros((self.array_size, self.dimensions))
    
        # apply respective biases to neurons in the population 
        J = np.array(state[self.bias])

        for ei in self.encoded_input.values():
            # add its values directly to the input current
            J += state[ei]

        # XXX decoded_input is missing
        # only do this if there is decoded_input
        if len(self.decoded_input) > 0:
            # add to input current for each neuron as
            # represented input signal x preferred direction
            J = [J[i] + np.dot(state[self.encoders][i], X[i].T)
                 for i in range(self.array_size)]

        # if noise has been specified for this neuron,
        if self.noise: 
            # generate random noise values, one for each input_current element, 
            # with standard deviation = sqrt(self.noise=std**2)
            # When simulating white noise, the noise process must be scaled by
            # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
            if self.noise.type == 'gaussian':
                J += state[self.s_rng].normal(
                    size=self.bias.shape, std=np.sqrt(self.noise.std / dt))
            elif self.noise.type == 'uniform':
                J += state[self.s_rng].uniform(
                    size=self.bias.shape, 
                    low=self.noise.low / np.sqrt(dt), 
                    high=self.noise.high / np.sqrt(dt))
        state[self.input_current] = J


def Ensemble(*args, **kwargs):
    if kwargs.pop('mode', 'spiking') == 'spiking':
        return SpikingEnsemble(*args, **kwargs)
    else:
        return DirectEnsemble(*args, **kwargs)

