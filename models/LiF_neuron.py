import numpy as np


class LIFNeuron:

    def __init__(self, K=1, decay=0.03, omega_rate=0.5, pre_x=1, pre_y=1):

        self.K = K
        self.decay = decay

        self.pre_syn = pre_x * pre_y
        self.omega_rate = omega_rate
        if omega_rate < 0:
            self.weight_m = np.ones((pre_x, pre_y))
        else:
            self.weight_m = np.random.rand(pre_x, pre_y) * omega_rate

        ### Momentum weight update
        self.update_prev = np.zeros((pre_x, pre_y))
        self.momentum = 0

        self.state = 0.0
        self.ref_time = 2.0
        self.ref_period = 0.0

    def event_input(self, x, y, values):

        if self.ref_period == 0:
            for i in range(0, x.__len__()):
                self.state += self.weight_m[int(x[i]), int(y[i])] * values[i]
        else:
            self.ref_period = self.ref_period - 1

    def decay_step(self):

        if self.state >= self.K:
            self.ref_period = self.ref_time
            self.state = 0
            return 1
        else:
            delta_state = self.state * self.decay
            self.state = self.state - delta_state
            return 0

    def clear(self):
        self.state = 0
        self.ref_period = 0

    ### Training functions
    def STDP_weight_update(self, pre_out, post_out):
        i = 0