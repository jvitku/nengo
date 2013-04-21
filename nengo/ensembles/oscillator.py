title='Oscillator'
label='Oscillator'
icon='oscillator.png'

description="""<html>The oscillator needs an input (2D), which can be used to start it off from rest.  Once it has started, the input can be set to zero. The controlled oscillator also needs a 1D control input that changes the frequency (and direction) of the oscillation. <a href="http://nengo.ca/docs/html/demos/oscillator.html">Tips & tricks.</a></html>"""

params=[
    ('name','Name',str,'Name of the oscillator'),
    ('neurons','Number of neurons',int,'Number of neurons in the oscillator'),
    ('frequency','Oscillator frequency (Hz)',float,'The speed at which the osillator oscillates'),
    ('tau_feedback','Feedback time constant [s]',float,'Synaptic time constant of the oscillator feedback, in seconds (longer -> slower change but better value retention)'),
    ('tau_input','Input PSTC [s]',float,'Post-synaptic time constant of the oscillator input, in seconds (longer -> more input filtering)'),
    ('scale','Scaling factor',float,'A scaling value for the input'),
    ('controlled', 'Frequency control', bool,'If checked will build a frequency controlled oscillator (a nonlinear system)'),
    ]


import numeric
def feedback(x):
    return x[0]+x[2]*x[1], x[1]-x[2]*x[0], 0
    
from java.util import ArrayList

def make(network, name='Oscillator', neurons=100, dimensions=2, frequency=5, 
         tau_feedback=0.1, tau_input=0.01, scale=1, controlled=False):
        """The oscillator needs an input (2D), which can be used to start it 
        off from rest.  Once it has started, the input can be set to zero. 
        The controlled oscillator also needs a 1D control input that changes 
        the frequency (and direction) of the oscillation. 
        http://nengo.ca/docs/html/demos/oscillator.html Tips & tricks.
    
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
    
    frequency = frequency*2*math.pi;
    if (controlled):
        oscillator=net.make(name,neurons,dimensions=3)
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        B = [[tau_input*scale, 0], [0, tau_input*scale], [0, 0]] 
        net.connect(oscillator, oscillator, A, func=feedback, pstc=tau_feedback)
        oscillator.addDecodedTermination('frequency', [[0],[0],[tau_feedback*frequency]], tau_input, False)
             
    else:
        oscillator=net.make(name, neurons, dimensions=2)
        A = [[1, -frequency*tau_feedback], [frequency*tau_feedback, 1]]
        B = [[1, 0],[0, 1]]
        net.connect(oscillator, oscillator, A, pstc=tau_feedback)

    oscillator.addDecodedTermination('input', B, tau_input, False)

    
    if net.network.getMetaData("oscillator") == None:
        net.network.setMetaData("oscillator", HashMap())
    oscillators = net.network.getMetaData("oscillator")

    oscillator=HashMap(8)
    oscillator.put("name", name)
    oscillator.put("neurons", neurons)
    oscillator.put("dimensions", dimensions)
    oscillator.put("frequency", frequency)
    oscillator.put("tau_feedback", tau_feedback)
    oscillator.put("tau_input", tau_input)
    oscillator.put("scale", scale)
    oscillator.put("controlled", controlled)

    oscillators.put(name, oscillator)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(name)

    if net.network.getMetaData("templateProjections") == None:
        net.network.setMetaData("templateProjections", HashMap())
    templateproj = net.network.getMetaData("templateProjections")
    templateproj.put(name, name)
    
def check_parameters(net,p):
    try:
       net.network.getNode(p['name'])
       return 'That name is already taken'
    except:
        pass
    if p['neurons']<1: return 'Must have a positive number of neurons'
    if p['frequency']<1: return 'Must have a positive frequency'
    
