try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict

class SymbolicSignal(object):
    def __init__(self, name=None, type=None, shape=None, units=None, dtype=None):
        slef.name = name
        self.type = type
        self.shape = shape
        self.units = units
        self.dtype = dtype


class Component(object):

    reset_inputs = ()
    reset_output = ()

    step_inputs = ()
    step_outputs = ()


def is_component(self, obj):
    return isinstance(obj, Component)


class find_components(object):
    def __init__(self, obj):
        self.lst = []
        self.lst_set = set()
        self.seen_ids = set()
        self.rec(obj)

    def rec(self, obj):
        if id(obj) in self.seen_ids:
            return

        if is_component(obj):
            self.lst.append(obj)
            self.lst_set.add(id(obj))

        self.seen_ids.add(id(obj))

        if isinstance(obj,
                (float,
                int,
                basestring,
                np.ndarray,
                type,
                function,
                )):
            return
        elif isinstance(obj, (list, tuple)):
            [self.rec(oo, lst) for oo in obj]
        elif isinstance(obj, (dict, OrderedDict)):
            for k, v in obj.items():
                self.rec(v, lst)
        elif isinstance(obj, object):
            for k in dir(obj):
                self.rec(getattr(obj, k), lst)
        else:
            raise TypeError()

def toposort_components(lst):
    return lst


class Simulator(object):
    def __init__(self, model, dt):
        self.model = model
        self.state = {}
        components = find_components(model).lst
        self.components = toposort_components(components)

    def __call__(self, simtime):
        n_steps = int(simtime / self.dt)
        dt = self.dt
        state = self.state
        for ii in xrange(n_steps):
            for cc in self.components:
                cc.update(state, dt)

