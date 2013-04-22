import copy
from functools import partial

import base

build_registry = {}
reset_registry = {}
step_registry = {}
probe_registry = {}

def register(cls, reg):
    def deco(f):
        reg[cls] = f
        return f
    return deco

register_build = partial(register, reg=build_registry)
register_reset = partial(register, reg=reset_registry)
register_step = partial(register, reg=step_registry)
register_probe = partial(register, reg=probe_registry)


class Simulator(base.Simulator):
    def __init__(self, network, dt, verbosity=0):
        base.Simulator.__init__(self, network)
        self.state = {}
        self.dt = dt
        self.verbosity = verbosity
        self.state[base.simulation_time] = self.simulation_time
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
        base.Simulator.reset(self)
        self.state[base.simulation_time] = self.simulation_time
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
            new_state[base.simulation_time] = self.simulation_time
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


# TimeNode implementation

@register_build(base.TimeNode)
def time_node_build(node, state, dt):
    state[node.output] = node.func(state[base.simulation_time])

@register_reset(base.TimeNode)
def time_node_reset(node, state):
    state[node.output] = node.func(state[base.simulation_time])

@register_step(base.TimeNode)
def time_node_step(node, old_state, new_state):
    t = new_state[base.simulation_time]
    new_state[node.output] = node.func(t)


# Probe implementation

@register_reset(base.Probe)
def probe_reset(probe, state):
    val0 = copy.deepcopy(state[probe.target])
    state[probe.stats] = [val0]

@register_step(base.Probe)
def probe_step(probe, old_state, new_state):
    stats = old_state[probe.stats]
    stats.append(copy.deepcopy(new_state[probe.target]))
    new_state[probe.stats] = stats

@register_probe(base.Probe)
def probe(probe, state):
    rval = state[probe.stats]
    state[probe.stats] = []
    return rval



class PureUniform(base.Uniform):
    def draw(self, size):
        return [random.uniform(self.low, self.high)
                for ii in xrange(size)]


class PureGaussian(base.Gaussian):
    def draw(self, size):
        return [random.gaussian(self.mean, self.std)
                for ii in xrange(size)]


