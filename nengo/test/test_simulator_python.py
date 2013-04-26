import nose.tools
from math import sin
from nengo.object_api import (
    Connection,
    LIFNeurons,
    Network,
    Probe,
    simulation_time,
    SelfDependencyError,
    Simulator,
    TimeNode,
    )


def test_smoke_1():
    net = Network()
    tn = net.add(TimeNode(sin, name='sin'))
    net.add(Probe(tn.output))
    net.add(Probe(simulation_time))
    sim = Simulator(net, dt=0.001, verbosity=0)
    results = sim.run(.1)

    assert len(results[simulation_time]) == 101
    for i, t in enumerate(results[simulation_time]):
        assert results[tn.output][i] == sin(t)


def test_smoke_2():
    net = Network()
    net.add(Probe(simulation_time))

    ens = net.add(LIFNeurons(13))
    net.add(Probe(ens.output))

    sim = Simulator(net, dt=0.001, verbosity=0)
    results = sim.run(.1)

    assert len(results[simulation_time]) == 101
    total_n_spikes = 0
    for i, t in enumerate(results[simulation_time]):
        output_i = results[ens.output][i]
        assert len(output_i) == 13
        assert all(oi in (0, 1) for oi in output_i)
        total_n_spikes += sum(output_i)
    assert total_n_spikes > 0


def test_schedule_self_dependency():
    net = Network()
    net.add(Probe(simulation_time))

    ens1 = net.add(LIFNeurons(3))
    ens2 = net.add(LIFNeurons(5))
    c1 = net.add(Connection(ens1.output, ens2.input_current))
    c2 = net.add(Connection(ens2.output, ens1.input_current))

    nose.tools.assert_raises(SelfDependencyError,
        Simulator, net, dt=0.001, verbosity=0)


def test_schedule():
    net = Network()

    ens1 = net.add(LIFNeurons(3))
    ens2 = net.add(LIFNeurons(5))
    c1 = net.add(Connection(ens1.output, ens2.input_current))
    probe = net.add(Probe(ens2.output))
    assert ens1.output is not None
    assert ens2.input_current is not None

    sim = Simulator(net, dt=0.001, verbosity=0)
    assert sim.member_ordering == [ens1, c1, ens2, probe]

