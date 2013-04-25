
from math import sin
from nengo.object_api import (
        LIFNeurons,
        Network,
        NeuronEnsemble,
        Probe,
        simulation_time,
        Simulator,
        TimeNode,
        )


def test_smoke_1():
    net = Network()
    tn = net.add(TimeNode(sin, name='sin'))
    net.add(Probe(tn.output))
    net.add(Probe(simulation_time))
    sim = Simulator(net, dt=0.001, verbosity=0, backend='numpy')
    results = sim.run(.1)

    assert len(results[simulation_time]) == 101
    for i, t in enumerate(results[simulation_time]):
        assert results[tn.output][i] == sin(t)


def test_smoke_2():
    net = Network()
    net.add(Probe(simulation_time))

    ens = net.add(NeuronEnsemble(LIFNeurons(13), dimensions=1))
    net.add(Probe(ens.spikes))

    sim = Simulator(net, dt=0.001, verbosity=0, backend='numpy')
    results = sim.run(.1)

    assert len(results[simulation_time]) == 101
    total_n_spikes = 0
    for i, t in enumerate(results[simulation_time]):
        output_i = results[ens.spikes][i]
        assert len(output_i) == 13
        assert all(oi in (0, 1) for oi in output_i)
        total_n_spikes += sum(output_i)
    assert total_n_spikes > 0


def test_smoke_3():
    # Learning!
    net = Network()
    net.add(Probe(simulation_time))

    ens1 = net.add(NeuronEnsemble(LIFNeurons(13), dimensions=1))
    ens2 = net.add(NeuronEnsemble(LIFNeurons(11), dimensions=1))
    conn = net.add(hPES_Connection(ens1, ens2, )

    net.add(Probe(ens.spikes))

    sim = Simulator(net, dt=0.001, verbosity=0, backend='numpy')
    results = sim.run(.1)

    assert len(results[simulation_time]) == 101
    total_n_spikes = 0
    for i, t in enumerate(results[simulation_time]):
        output_i = results[ens.spikes][i]
        assert len(output_i) == 13
        assert all(oi in (0, 1) for oi in output_i)
        total_n_spikes += sum(output_i)
    assert total_n_spikes > 0
