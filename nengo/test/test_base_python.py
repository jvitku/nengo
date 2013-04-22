from math import sin
from nengo.base import (
        TimeNode,
        Network,
        Probe,
        simulation_time
        )
from nengo.base_python import Simulator


def test_1():
    net = Network()
    tn = net.add(TimeNode(sin, name='sin'))
    net.add(Probe(tn.output))
    net.add(Probe(simulation_time))
    sim = Simulator(net, dt=0.001, verbosity=1)
    results = sim.run(.1)

    assert len(results[simulation_time]) == 101
    for i, t in enumerate(results[simulation_time]):
        assert results[tn.output][i] == sin(t)

