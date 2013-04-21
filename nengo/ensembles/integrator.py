from ..connection import gen_transform
from ..networks import array

def make(network, input, name='Integrator', neurons=100, dimensions=1, 
         tau_feedback=0.1, tau_input=0.01, scale=1):
    """This constructs an integrator of the specified number of dimensions. 
    It requires an input of that number of dimensions after construction.
    
    :param str name:
        Name of the integrator
    :param int neurons:
        Number of neurons in the integrator
    :param int dimensions:
        Number of dimensions for the integrator
    :param float tau_feedback:
         Post-synaptic time constant of the integrative feedback, 
         in seconds (longer -> slower change but better value retention)
    :param float tau_input:
        Post-synaptic time constant of the integrator input, in seconds 
        (longer -> more input filtering)
    :param float scale:
        A scaling value for the input (controls the rate of integration)
    """
    net = _check_parameters(network, name, neurons, dimensions, tau_feedback, 
                           tau_input, scale)
    
    if (dimensions<8):
        integrator=net.make(name, neurons, dimensions)
    else:
        integrator=array.make(name, int(neurons/dimensions), dimensions)
    
    recurrent_connection = net.connect(integrator, integrator, 
                                       pstc=tau_feedback)
    
    eye=gen_transform(dimensions, dimensions)
    input_connection = net.connect(input, integrator, eye*tau_feedback*scale, 
                                   tau_input)
    
    return integrator, recurrent_connection, input_connection

def _check_parameters(network, name, neurons, dimensions, tau_feedback, 
                     tau_input, scale):
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