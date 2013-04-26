import copy
from functools import partial

import numpy as np

import object_api as API
import simulator_python
from simulator_python import ImplBase


class Simulator(simulator_python.Simulator):

    # These classes create a new registry (distinct from parent class)
    impl_registry = {}
    build_registry = {}
    reset_registry = {}
    step_registry = {}
    probe_registry = {}
    sample_registry = {}

    def __init__(self, network, dt, verbosity=0):
        simulator_python.Simulator.__init__(self, network, dt, verbosity)

API.SimulatorBase._backends['numpy'] = Simulator
register_impl = Simulator.register_impl


@register_impl
class TimeNode(ImplBase):
    @staticmethod
    def build(node, state, dt):
        state[node.output] = np.asarray(node.func(state[API.simulation_time]))

    @staticmethod
    def reset(node, state):
        state[node.output] = np.asarray(node.func(state[API.simulation_time]))

    @staticmethod
    def step(node, old_state, new_state):
        t = new_state[API.simulation_time] # XXX new state or old_state?
        new_state[node.output] = np.asarray(node.func(t))


register_impl(simulator_python.Probe)


def draw(dist, rng, N):
    if dist.dist_name == 'uniform':
        return rng.uniform(dist.low, dist.high, size=N)
    elif dist.dist_name == 'gaussian':
        return rng.normal(mu=dist.mean, std=dist.std, size=N)
    else:
        raise NotImplementedError()


@register_impl
class LIFNeurons(ImplBase):
    @staticmethod
    def build(neurons, state, dt):
        r1 = np.random.RandomState(neurons.max_rate.seed)
        r2 = np.random.RandomState(neurons.intercept.seed)

        max_rates = draw(neurons.max_rate, r1, neurons.size)
        threshold = draw(neurons.intercept, r2, neurons.size)
        
        u = neurons.tau_ref - (1.0 / max_rates)
        x = 1.0 / (1 - np.exp(u / neurons.tau_rc))
        alpha = (1 - x) / threshold
        j_bias = 1 - alpha * threshold 

        state[neurons.alpha] = alpha
        state[neurons.j_bias] = j_bias
        LIFNeurons.reset(neurons, state)

    @staticmethod
    def reset(neurons, state):
        state[neurons.voltage] = np.zeros(neurons.size)
        state[neurons.refractory_time] = np.zeros(neurons.size)
        state[neurons.output] = np.zeros(neurons.size)

    @staticmethod
    def step(neurons, old_state, new_state):
        alpha = old_state[neurons.alpha]
        j_bias = old_state[neurons.j_bias]
        # XXX why this one should be new_state?
        if neurons.input_current in new_state:
            J  = j_bias + new_state[neurons.input_current]
        else:
            J  = j_bias 
        voltage = old_state[neurons.voltage]
        refractory_time = old_state[neurons.refractory_time]
        tau_rc = neurons.tau_rc
        tau_ref = neurons.tau_ref
        dt = new_state[API.simulation_time] - old_state[API.simulation_time]
        # -- use the input_current that was computed by the connections
        # -- on the last time through

        # Euler's method
        dV = dt / tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked = (v > 1)

        # adjust refractory time (neurons that spike get
        # a new refractory time set, all others get it reduced by dt)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV
        spiketime = dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        new_refractory_time = (
            spiked * (spiketime + tau_ref)
            + (1 - spiked) * (refractory_time - dt))

        new_voltage = v * (1 - spiked)
        new_output = spiked


        new_state[neurons.alpha] = alpha
        new_state[neurons.j_bias] = j_bias
        new_state[neurons.voltage] = new_voltage
        new_state[neurons.refractory_time] = new_refractory_time
        new_state[neurons.output] = new_output


@register_impl
class Connection(ImplBase):
    @staticmethod
    def reset(self, state):
        src = state[self.inputs['X']]
        dst = src.copy()
        state[self.outputs['X']] = dst

    @staticmethod
    def step(self, old_state, new_state):
        src = new_state[self.inputs['X']]
        dst = src.copy()
        new_state[self.outputs['X']] = dst


@register_impl
class hPES_Connection(ImplBase):
    @staticmethod
    def build(self, state, dt):
        encoders = state[self.dst.encoders]
        src_filtered = state[self.src.spikes].copy()
        dst_filtered = state[self.dst.spikes].copy()
        state[self.src_filtered] = src_filtered
        state[self.dst_filtered] = dst_filtered

    @staticmethod
    def reset(self, state):
        encoders = state[self.dst.encoders]
        rng = np.random.RandomState(self.seed)
        theta = rng.uniform(low=5e-5, high=15e-5,
                            size=(self.dst.array_size, self.dst.neurons_num))
        gains = np.sqrt((encoders ** 2).sum(axis=-1))
        theta *= gains
        state[self.gains] = gains
        state[self.theta] = theta

    @staticmethod
    def tick(self, old_state, new_state):
        encoders = old_state[self.encoders] # XXX new_state or old_state
        error_signal = old_state[self.error_signal] # vector??
        supervised_rate = old_state[self.supervised_learning_rate]
        unsupervised_rate = supervised_rate * self.unsupervised_rate_factor

        encoded_error = (encoders * error_signal[newaxis, :]).sum(axis=-1)


        delta_supervised = (supervised_rate
                            * src_filtered[None,:]
                            * encoded_error[:,None])

        delta_unsupervised = (unsupervised_rate
                              * self.src_filtered[None,:]
                              * self.dst_filtered[:,None]
                              * (self.dst_filtered-self.theta)[:,None]
                              * self.gains[:,None])

        new_wm = (weight_matrix
                + self.supervision_ratio * delta_supervised
                + (1 - self.supervision_ratio) * delta_unsupervised)

        # update filtered inputs
        dt = new_state[simulation_time] - old_state[simulation_time]
        alpha = dt / self.pstc
        new_src = src_filtered + alpha * (src_spikes - src_filtered)
        new_dst = dst_filtered + alpha * (dst_spikes - dst_filtered)

        # update theta
        alpha_theta = dt / self.theta_tau
        new_theta = theta + alpha_theta * (new_dst - theta)

        state[self.weight_matrix] =  new_wm
        state[self.src_filtered] = new_src
        state[self.dst_filtered] = new_dst
        state[self.theta] = new_theta


@register_impl
class Filter(ImplBase):
    @staticmethod
    def reset(self, state):
        state[self.output] = np.zeros(self.output.size)

    @staticmethod
    def step(self, old_state, new_state):
        new_state[self.output] = (
            old_state[self.output]
            + self.tau * old_state[self.inputs['var']])


@register_impl
class LinearNeurons(ImplBase):
    @staticmethod
    def reset(self, state):
        state[self.outputs['X']] = np.array(
            state[self.inputs['input_current']])

    @staticmethod
    def step(self, old_state, new_state):
        state[self.outputs['X']] = np.array(
            state[self.inputs['input_current']])


@register_impl
class MSE_MinimizingConnection(ImplBase):
    @staticmethod
    def reset(self, state):
        state[self.outputs['X']] = np.asarray([0])
        state[self.outputs['error_signal']] = np.zeros_like(
                state[self.inputs['target']])

    @staticmethod
    def step(self, old_state, new_state):
        src = state[self.inputs['X']])
        target = state[self.inputs['target']])

        state[self.outputs['X']] = np.asarray([0])
        state[self.outputs['error_signal']] = np.zeros_like(
                state[self.inputs['target']])




class NeuronEnsemble(ImplBase):
    @staticmethod
    def build(self, state, dt):
        # -- neurons are not known to the simulator, so build them directly
        build(self.neurons, state, dt)

        # N.B. re-use neurons rng
        rng = state[self.neurons.rng]

        encoders = rng.randn(self.neurons.size, self.dimensions)

        encoders = self.make_encoders(encoders=self.encoders)
        norm = np.sum(encoders * encoders, axis=1).reshape(
            (self.neuron_model.size, 1))
        encoders = encoders / np.sqrt(norm) * neuron_model.alpha[: None]

        state[self.encoders] = encoders

    @staticmethod
    def reset(self, state):
        reset(self.neurons, state)

    @staticmethod
    def step(self, old_state, new_state):
        
        step(self.neurons, old_state, new_state)
        # find the total input current to this population of neurons
        
        # apply respective biases to neurons in the population 
        J = np.zeros((self.neuron_model.size, 1))
        J += self.neuron_model.j_bias

        #add in neuron->neuron currents
        for c in self.neuron_inputs:
            # add its values directly to the input current
            J += c.get_post_input(old_state, dt)

        #add in vector->vector currents
        for c in self.vector_inputs:
            fuck = c.get_post_input(old_state, dt) 
            J += np.dot( self.encoders, 
                c.get_post_input(old_state, dt)).reshape(
                (self.neuron_model.size, 1))

        # if noise has been specified for this neuron,
        if self.noise: 
            # generate random noise values, one for each input_current element, 
            # with standard deviation = sqrt(self.noise=std**2)
            # When simulating white noise, the noise process must be scaled by
            # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
            if self.noise.type == 'gaussian':
                J += random.gaussian(
                    size=self.bias.shape, std=np.sqrt(self.noise/dt))
            '''elif self.noise.type == 'uniform':
                J += random.uniform(
                    size=self.bias.shape, 
                    low=-self.noise / np.sqrt(dt), 
                    high=self.noise / np.sqrt(dt))'''
        
        # pass the input current total into the neuron model
        self.spikes = self.neuron_model._step(new_state, J, dt)
    
        # update the weight matrices on learned terminations
        for c in self.vector_inputs+self.neuron_inputs:
            c.learn(dt)

        # compute the decoded origin decoded_input from the neuron output
        for i,o in enumerate(self.outputs):
            new_state[o] = np.dot(self.neuron_model.output, self.decoders[i])

