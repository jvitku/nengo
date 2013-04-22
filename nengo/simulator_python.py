import copy
from functools import partial
import math
import random

import object_api as API

build_registry = {}
reset_registry = {}
step_registry = {}
probe_registry = {}
sample_registry = {}

def register(cls, reg):
    def deco(f):
        reg[cls] = f
        return f
    return deco

def call_registry(obj, *args, **kwargs):
    reg = kwargs.pop('reg')
    return reg[type(obj)](obj, *args, **kwargs)

def registry_stuff(reg):
    a = partial(call_registry, reg=reg)
    b = partial(register, reg=reg)
    return a, b

build, register_build = registry_stuff(build_registry)
reset, register_reset = registry_stuff(reset_registry)
step, register_step = registry_stuff(step_registry)
probe, register_probe = registry_stuff(probe_registry)
sample, register_sample = registry_stuff(sample_registry)


def register_step_carry_forward(cls, *args):
    def carry_forward_step(obj, old_state, new_state):
        for arg in args:
            var = getattr(obj, arg)
            new_state[var] = old_state[var]
    register_step(cls)(carry_forward_step)


class Simulator(API.Simulator):
    def __init__(self, network, dt, verbosity=0):
        API.Simulator.__init__(self, network)
        self.state = {}
        self.dt = dt
        self.verbosity = verbosity
        self.state[API.simulation_time] = self.simulation_time
        for member in self.network.all_members:
            build_fn = build_registry.get(type(member), None)
            if build_fn:
                if verbosity:
                    print 'Build:', member, build_fn
                build_fn(member, self.state, self.dt)
            elif verbosity:
                print 'No build:', member
        self.reset()

    def reset(self):
        API.Simulator.reset(self)
        self.state[API.simulation_time] = self.simulation_time
        for member in self.network.all_members:
            reset_fn = reset_registry.get(type(member), None)
            if reset_fn:
                if self.verbosity:
                    print 'Reset:', member, reset_fn
                reset_fn(member, self.state)
            elif self.verbosity:
                print 'No reset:', member

    def run_steps(self, steps):
        old_state = self.state
        step_fns = []
        for member in self.network.all_members:
            step_fn = step_registry.get(type(member), None)
            if step_fn:
                if self.verbosity:
                    print 'Step:', member, step_fn
                step_fns.append((step_fn, member))
            elif self.verbosity:
                print 'No step:', member

        for tt in xrange(steps):
            self.simulation_time += self.dt
            new_state = {}
            new_state[API.simulation_time] = self.simulation_time
            for step_fn, member in step_fns:
                step_fn(member, old_state, new_state)
            old_state = new_state
        self.state = new_state

        rval = {}
        for probe in self.network.all_probes:
            # -- probe_fn is mandatory
            probe_fn = probe_registry[type(probe)]
            rval[probe.target] = probe_fn(probe, self.state)
        return rval

    def run(self, sim_time):
        steps = int(sim_time / self.dt)
        return self.run_steps(steps)


#
# TimeNode
#

@register_build(API.TimeNode)
def time_node_build(node, state, dt):
    state[node.output] = node.func(state[API.simulation_time])

@register_reset(API.TimeNode)
def time_node_reset(node, state):
    state[node.output] = node.func(state[API.simulation_time])

@register_step(API.TimeNode)
def time_node_step(node, old_state, new_state):
    t = new_state[API.simulation_time]
    new_state[node.output] = node.func(t)


#
# Probe
#

@register_reset(API.Probe)
def probe_reset(probe, state):
    val0 = copy.deepcopy(state[probe.target])
    state[probe.stats] = [val0]

@register_step(API.Probe)
def probe_step(probe, old_state, new_state):
    stats = old_state[probe.stats]
    stats.append(copy.deepcopy(new_state[probe.target]))
    new_state[probe.stats] = stats

@register_probe(API.Probe)
def probe(probe, state):
    rval = state[probe.stats]
    state[probe.stats] = []
    return rval

#
# Uniform, Gaussian
#

@register_build(API.Uniform)
def uniform_build(dist, state, dt):
    rng = random.Random(dist.seed)
    def draw_n(N):
        return[rng.uniform(dist.low, dist.high) for ii in xrange(N)]
    state[dist.rng] = draw_n

@register_build(API.Gaussian)
def gaussian_build(dist, state, dt):
    rng = random.Random(dist.seed)
    def draw_n(N):
        return[rng.gauss(dist.mean, dist.std) for ii in xrange(N)]
    state[dist.rng] = draw_n

register_step_carry_forward(API.Uniform, 'rng')
register_step_carry_forward(API.Gaussian, 'rng')


#
# LIFNeurons
#

@register_build(API.LIFNeurons)
def lifneurons_build(neurons, state, dt):
    build(neurons.max_rate, state, dt)
    build(neurons.intercept, state, dt)

    max_rates = state[neurons.max_rate.rng](neurons.size)
    threshold = state[neurons.intercept.rng](neurons.size)

    def x_fn(max_rate):
        u = neurons.tau_ref - (1.0 / max_rate)
        return 1.0 / (1 - math.exp(u / neurons.tau_rc))
    xlist = map(x_fn, max_rates)
    alpha = [(1 - x) / intercept for x, intercept in zip(xlist, threshold)]
    j_bias = [1 - aa * intercept for aa, intercept in zip(alpha, threshold)]

    state[neurons.alpha] = alpha
    state[neurons.j_bias] = j_bias
    state[neurons.voltage] = [0] * neurons.size
    state[neurons.refractory_time] = [0] * neurons.size
    state[neurons.output] = [0] * neurons.size

@register_reset(API.LIFNeurons)
def lifneurons_reset(neurons, state):
    state[neurons.voltage] = [0] * neurons.size
    state[neurons.refractory_time] = [0] * neurons.size
    state[neurons.output] = [0] * neurons.size

@register_step(API.LIFNeurons)
def lifneurons_step(neurons, old_state, new_state):
    alpha = old_state[neurons.alpha]
    j_bias = old_state[neurons.j_bias]
    voltage = old_state[neurons.voltage]
    refractory_time = old_state[neurons.refractory_time]
    tau_rc = neurons.tau_rc
    tau_ref = neurons.tau_ref
    dt = new_state[API.simulation_time] - old_state[API.simulation_time]
    J  = j_bias # XXX WRONG MATH
    new_voltage = [0] * neurons.size
    new_refractory_time = [0] * neurons.size
    new_output = [0] * neurons.size

    def clip(a, low, high):
        if a < low:
            return low
        if a > high:
            return high
        return a

    for ii in xrange(neurons.size):

        # Euler's method
        dV = dt / tau_rc * (J[ii] - voltage[ii])

        # increase the voltage, ignore values below 0
        v = max(voltage[ii] + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time[ii] - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        if post_ref < 0:
            v = 0
        elif post_ref < 1:
            v *= post_ref

        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked = 1 if v > 1 else 0

        # adjust refractory time (neurons that spike get
        # a new refractory time set, all others get it reduced by dt)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV
        spiketime = dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        if spiked:
            new_refractory_time[ii] = spiketime + tau_ref
        else:
            new_refractory_time[ii] = refractory_time[ii] - dt

        new_voltage[ii] = v * (1 - spiked)
        new_output[ii] = spiked


    new_state[neurons.alpha] = alpha
    new_state[neurons.j_bias] = j_bias
    new_state[neurons.voltage] = new_voltage
    new_state[neurons.refractory_time] = new_refractory_time
    new_state[neurons.output] = new_output


# NeuronEnsemble implementation
@register_build(API.NeuronEnsemble)
def neuron_ensemble_build(ens, state, dt):
    build(ens.neurons, state, dt)

    # compute encoders
    # TODO
    #self.encoders = self.make_encoders(encoders=encoders)
    # combine encoders and gain for simplification
    #self.encoders = (self.encoders.T * alpha.T).T

@register_reset(API.NeuronEnsemble)
def neuron_ensemble_reset(ens, state):
    reset(ens.neurons, state)

@register_step(API.NeuronEnsemble)
def neuron_ensemble_step(ens, old_state, new_state):
    step(ens.neurons, old_state, new_state)


class PureUniform(API.Uniform):
    def draw(self, size):
        return [random.uniform(self.low, self.high)
                for ii in xrange(size)]


class PureGaussian(API.Gaussian):
    def draw(self, size):
        return [random.gaussian(self.mean, self.std)
                for ii in xrange(size)]


