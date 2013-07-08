import numpy as np
from simulator_objects import Constant, Signal

#
# Definitions of standard kinds of Non-Linearity

# TODO: Consider moving these into simulator_objects.py
# because they are the basic objects that are destinations for
# e.g. encoders and sources for decoders. They are tightly
# part of that set of objects.
#

class Direct(object):
    """
    """
    def __init__(self, n_in, n_out, fn, name=None):
        """
        fn:
        """
        if name is None:
            self.input_signal = Signal(n_in)
            self.output_signal = Signal(n_out)
            self.bias_signal = Constant(n=n_in, value=np.zeros(n_in))
        else:
            self.input_signal = Signal(n_in,
                                      name=name + '.input')
            self.output_signal = Signal(n_out,
                                       name=name + '.output')
            self.bias_signal = Constant(n=n_in,
                                        value=np.zeros(n_in),
                                       name=name + '.bias')

        self.n_in = n_in
        self.n_out = n_out
        self.fn = fn

    def __str__(self):
        return "Direct (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def fn(self, J):
        return J


class LIF(object):
    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, upsample=1,
                name=None):
        if name is None:
            self.input_signal = Signal(n_neurons)
            self.output_signal = Signal(n_neurons)
            self.bias_signal = Constant(n=n_neurons, value=np.zeros(n_neurons))
        else:
            self.input_signal = Signal(n_neurons, name=name + '.input')
            self.output_signal = Signal(n_neurons, name=name + '.output')
            self.bias_signal = Constant(n=n_neurons,
                                        value=np.zeros(n_neurons),
                                       name=name + '.bias')

        self.n_neurons = n_neurons
        self.upsample = upsample
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gain = np.random.rand(n_neurons)

    def __str__(self):
        return "LIF (id " + str(id(self)) + ", " + str(self.n_neurons) + "N)"

    def __repr__(self):
        return str(self)

    @property
    def bias(self):
        return self.bias_signal.value

    @bias.setter
    def bias(self, value):
        self.bias_signal.value[...] = value

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons

    def set_gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons

        """
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # N.B. J here *includes* bias

        # Euler's method
        dV = dt / self.tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked[:] = (v > 1) * 1.0

        old = np.seterr(all='ignore')
        try:

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            new_refractory_time = spiked * (spiketime + self.tau_ref) \
                                  + (1 - spiked) * (refractory_time - dt)
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        voltage[:] = v * (1 - spiked)
        refractory_time[:] = new_refractory_time

    def rates(self, J_without_bias):
        """Return LIF firing rates for current J in Hz

        Parameters
        ---------
        J: ndarray of any shape
            membrane voltages
        tau_rc: broadcastable like J
            XXX
        tau_ref: broadcastable like J
            XXX
        """
        old = np.seterr(all='ignore')
        J = J_without_bias + self.bias
        try:
            A = self.tau_ref - self.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A


class LIFRate(LIF):
    def __init__(self, n_neurons):
        LIF.__init__(self, n_neurons)
        self.input_signal = Signal(n_neurons)
        self.output_signal = Signal(n_neurons)
        self.bias_signal = Constant(n=n_neurons, value=np.zeros(n_neurons))

    def __str__(self):
        return "LIFRate (id " + str(id(self)) + ", " + str(self.n_neurons) + "N)"

    def __repr__(self):
        return str(self)

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons
