title='Integrator'
label='Integrator'
icon='integrator.png'

description="""<html>This constructs an integrator of the specified number of dimensions. It requires an input of that number of dimensions after construction.</html>"""

params=[
    ('name','Name',str,'Name of the integrator'),
    ('neurons','Number of neurons',int,'Number of neurons in the integrator'),
    ('dimensions','Number of dimensions',int,'Number of dimensions for the integrator'),
    ('tau_feedback','Feedback PSTC [s]',float,'Post-synaptic time constant of the integrative feedback, in seconds (longer -> slower change but better value retention)'),
    ('tau_input','Input PSTC [s]',float,'Post-synaptic time constant of the integrator input, in seconds (longer -> more input filtering)'),
    ('scale','Scaling factor',float,'A scaling value for the input (controls the rate of integration)'),
    ]
    
import numeric
from java.util import ArrayList
from java.util import HashMap

def make(network, name='Integrator', neurons=100, dimensions=1, tau_feedback=0.1, tau_input=0.01, scale=1):
    net = check_parameters(network, name, neurons, dimensions, tau_feedback, tau_input, scale)
    
    if (dimensions<8):
        integrator=net.make(name,neurons,dimensions)
    else:
        integrator=net.make_array(name, int(neurons/dimensions),dimensions, quick=True)
    net.connect(integrator,integrator,pstc=tau_feedback)
    integrator.addDecodedTermination('input',numeric.eye(dimensions)*tau_feedback*scale,tau_input,False)
    if net.network.getMetaData("integrator") == None:
        net.network.setMetaData("integrator", HashMap())
    integrators = net.network.getMetaData("integrator")

    integrator=HashMap(6)
    integrator.put("name", name)
    integrator.put("neurons", neurons)
    integrator.put("dimensions", dimensions)
    integrator.put("tau_feedback", tau_feedback)
    integrator.put("tau_input", tau_input)
    integrator.put("scale", scale)

    integrators.put(name, integrator)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(name)

    if net.network.getMetaData("templateProjections") == None:
        net.network.setMetaData("templateProjections", HashMap())
    templateproj = net.network.getMetaData("templateProjections")
    templateproj.put(name, name)

    integrator.addDecodedTermination('input', numeric.eye(dimensions)*tau_feedback*scale, tau_input, False)

def check_parameters(network, name, neurons, dimensions, tau_feedback, tau_input, scale):
    if isinstance(network, str):
        net = Model.get(network, None)
        if net is None:
            raise ValueError("%s doesn't exist, can't add integrator" % network)
    else:
        net = network
    if not isinstance(network, Network):
        raise valueError("'Network' is not a Network object") 
    if network.get(name, None) is not None:
        raise ValueError("That name is already taken in this network")

    if neurons < 1: raise ValueError('Must have a positive number of neurons')
    if dimensions < 1: raise ValueError('Must have at least one dimension')
    
    return net