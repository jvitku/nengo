import nef
from ca.nengo.ui.configurable import *
from javax.swing import *
from javax.swing.event import DocumentListener

title='Network Array'
label='Network\nArray'
icon='array.png'

description="""<html>This template enables constructing subnetworks full D (# of dimensions) independent populations of neurons.  These are faster to construct but cannot compute all the same nonlinear functions as a single large population with D dimensions.</html>"""

params=[
    ('name','Name',str, 'Name of the Network Array'),
    ('neurons','Neurons per dimension',int,'Number of neurons in each of the ensembles'),
    ('length','Number of dimensions',int,'Number of ensembles in the array'),
    ('radius','Radius',float,'Maximum magnitude of vector that can be represented in each ensemble'),
    ('iLow','Intercept (low)',float,'Smallest value for neurons to start firing at (between -1 and 1)'),
    ('iHigh','Intercept (high)',float,'Largest value for neurons to start firing at (between -1 and 1)'),
    ('rLow','Max rate (low) [Hz]',float,'Smallest maximum firing rate for neurons in the ensemble'),
    ('rHigh','Max rate (high) [Hz]',float,'Largest maximum firing rate for neurons in the ensemble'),
    ('encSign','Encoding sign', PTemplateSign,'Limits the sign of the encoders'),
    ('useQuick', 'Quick mode', bool,'Uses the exact same encoders and decoders for each ensemble in the array'),
    ]


from ..nef import Model

def make(network, name, ensembles, neurons, dimensions, **ensemble_params):
         # radius=1.0, rLow=200, rHigh=400, iLow=-1, iHigh=1, encSign=0, useQuick=True):
    network = Model.get(network)

    check_parameters(network)

    if encSign!=0:
        ensemble = net.make_array(name, neurons, length, max_rate=(rLow,rHigh), intercept=(iLow, iHigh), radius=radius, encoders=[[encSign]], quick=useQuick)
    else:
        ensemble = net.make_array(name, neurons, length, max_rate=(rLow,rHigh), intercept=(iLow, iHigh), radius=radius, quick=useQuick)

def check_parameters(network):
    try:
       net.network.getNode(p['name'])
       return 'That name is already taken'
    except:
        pass
    if p['iLow'] > p['iHigh']: return 'Low intercept must be less than high intercept'
    if p['rLow'] > p['rHigh']: return 'Low max firing rate must be less than high max firing rate'

