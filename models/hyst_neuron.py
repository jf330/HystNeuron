import numpy as np


class HystNeuron:

    def __init__(self, h=100, K=1, eta=0.95, a=1, b=0.5, omega_rate=0.02, pre_x=1, pre_y=1):

        ### ODEs parameters
        self.h = h
        self.K = K

        self.eta = eta
        self.a = a
        self.b = b

        ### Neuron state variables
        self.state = 0.0
        self.out = 0.0
        self.reset = 0.0

        self.pre_syn = pre_x * pre_y
        self.omega_rate = omega_rate
        if omega_rate < 0:
            self.weight_m = np.ones((pre_x,pre_y))
        else:
            self.weight_m = np.random.rand(pre_x,pre_y) * omega_rate
            # self.weight_m = np.random.randn(pre_x,pre_y) * omega_rate

        ### Momentum weight update
        self.update_prev = np.zeros((pre_x,pre_y))
        self.momentum = 0

    def clear(self):
        self.out = 0
        self.state = 0
        self.reset = 0

    def cont_current(self, current=0.05):
        self.state += current

    def event_input(self, x, y, values):
        for i in range(0, x.__len__()):
            self.state += self.weight_m[int(x[i]), int(y[i])] * values[i]

    def frame_input(self, frame):
        input = sum(frame * self.weight_m)
        self.state += input

    def decay_step(self):
        ### State decay function  FIXME Should delta_state or delta_reset be calculated first
        delta_state = (self.reset * self.eta * self.state) + ((1 - self.eta) * self.a * self.state)
        self.state = self.state - delta_state

        ### Different differentiable threshold implementations
        delta_reset = np.heaviside((self.state - self.K), self.K) - self.b * self.reset
        # delta_reset = (np.float_power(self.state, self.h) / (np.float_power(self.K, self.h) + np.float_power(self.state, self.h))) - self.b * self.reset
        # delta_reset = ((self.state**self.h) / (self.K**self.h + self.state**self.h)) - self.b * self.reset
        # delta_reset = (0.5 * (1 + np.tanh(self.h * (self.state - self.K)))) - self.b * self.reset

        self.reset = self.reset + delta_reset
        self.reset = np.clip(self.reset, 0, 1)

        return delta_state


    ### Training functions
    def STDP_weight_update(self, pre_out, post_out):
        i = 0

    def feedback_weight_update(self, output, data, error, error_trace, to_update=0.2, lr=0.001):

        ### Select top x% most eligible pre-syn neurons over whole input
        update_partition = np.rint((self.weight_m.__len__()) * to_update).astype(int)

        ### Correlation to post-syn output
        elig = []
        for i in range(0, len(self.weight_m)):
            elig.append(np.array(data[:, i]) * np.array(output))

        ### Correlation to error_trace
        # elig = []
        # for i in range(0, len(self.weight_m)):
            # elig.append(np.array(data[:, i]) * np.array(abs(error_trace)))
            # elig.append(np.array(data[:, i]) * np.array(error_trace))
            # elig.append(np.array(data[:, i]) * np.array(error_trace) * np.array(output))

        elig_sum = np.sum(elig, axis=1)
        most_elig_syn = np.argpartition(elig_sum, -update_partition)[-update_partition:]
        # print(-elig_sum[0])

        ### Update all blame-wise
        # update_new = []
        # for i in range(0, len(self.weight_m)):
        #     update_new.append(-elig_sum[i] * lr)

        ### Update most_elig_syn blame-wise
        update_new = []
        for i in range(0, len(self.weight_m)):
            if i in most_elig_syn:
                update_new.append(error * -lr)
                # if error > 0:
                #     update_new.append(-lr)
                # elif error < 0:
                #     update_new.append(lr)
                # else:
                #     update_new.append(0)
            else:
                update_new.append(0)

        update_all = np.array(update_new) + (self.update_prev * self.momentum)
        self.weight_m = self.weight_m + update_all
        # self.weight_m = np.clip(self.weight_m, 0, 1)

        self.update_prev = np.array(update_all)

        return elig

    def event_weight_update(self, output, x, y, error_trace, to_update=0.2, lr=0.001):

        ### Correlation to error_trace
        elig = np.zeros_like(self.weight_m)
        # for i in range(0, len(self.weight_m)):
            # elig.append(np.array(data[:, i]) * np.array(abs(error_trace)))
            # elig.append(np.array(data[:, i]) * np.array(error_trace))
            # elig.append(np.array(data[:, i]) * np.array(error_trace) * np.array(output))

        elig_sum = np.sum(elig, axis=1)

        ### Update all blame-wise
        update_new = []
        for i in range(0, len(self.weight_m)):
            update_new.append(-elig_sum[i] * lr)

        update_all = np.array(update_new) + (self.update_prev * self.momentum)
        self.weight_m = self.weight_m + update_all
        # self.weight_m = np.clip(self.weight_m, 0, 1)

        self.update_prev = np.array(update_all)

        return elig
