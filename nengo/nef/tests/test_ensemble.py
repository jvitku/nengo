import sys
from nengo.nef.neuron import LIFNeuron
from nengo.nef.ensemble import DirectEnsemble
from nengo.nef.ensemble import SpikingEnsemble
from nengo.nef.ensemble import Ensemble
from nengo.nef.simulator import Simulator

def test_smoke1():
    LIFNeuron(12)

def test_SpikingEnsemble():
    SpikingEnsemble(10, 5, neuron_model=LIFNeuron(10))

def test_DirectEnsemble():
    DirectEnsemble(10)

def test_Ensemble():
    Ensemble(neurons=40, dimensions=5, mode='spiking')

def test_Simulator():
    ens = Ensemble(neurons=40, dimensions=5, mode='spiking')

    class Model(object):
        def __init__(self):
            self.all_ensembles = [ens]

    sim = Simulator(Model(), dt=.001)
    sim(100)
    print sim.state




