

class PureLIFNeuron(LIFNeuron):

    def _pure_build(self, state, dt):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.
        
        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        
        """
        random.uniform
        max_rates = self.rng.uniform(size=self.size,
            low=max_rate[0], high=max_rate[1])  

        x = 1.0 / (1 - np.exp(
                (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        alpha = (1 - x) / (intercepts - 1.0)
        j_bias = 1 - alpha * intercepts
        return alpha, j_bias
