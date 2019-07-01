import numpy as np


class LIFNeuron:

    def __init__(self, K=1, decay=0.1, ref_time=2.0, omega_rate=0.5, pre_x=1, pre_y=1):

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
        self.momentum = 0.3

        self.state = 0.0
        self.ref_time = ref_time
        self.ref_period = 0.0

    def event_input(self, x, y, values):

        if self.ref_period == 0:
            for i in range(0, x.__len__()):
                self.state += self.weight_m[int(x[i]), int(y[i])] * values[i]

    def decay_step(self):

        if self.state >= self.K:
            self.ref_period = self.ref_time
            self.state = 0
            return 1
        else:
            delta_state = self.state * self.decay
            self.state = self.state - delta_state
            if self.ref_period != 0:
                self.ref_period = self.ref_period - 1
            return 0

    def clear(self):
        self.state = 0
        self.ref_period = 0

    ### Training functions
    def STDP_weight_update(self, pre_out, post_out):
        print("TODO")

    def feedback_weight_update(self, post, pre, error, error_trace, to_update=0.2, lr=0.001):

        ### Select top x% most eligible pre-syn neurons over whole input
        update_partition = np.rint((self.weight_m.__len__()) * to_update).astype(int)

        ### Correlation-based eligibility trace towards post-syn signal
        elig = []
        for i in range(0, len(self.weight_m)):
            elig.append(np.array(pre[:, i]) * np.array(post))
            # elig.append(np.array(pre[:, i]) * np.array(abs(error_trace)))
            # elig.append(np.array(pre[:, i]) * np.array(error_trace))
            # elig.append(np.array(pre[:, i]) * np.array(post) * np.array(error_trace))
            # elig.append(np.array(pre[:, i]) * np.array(post) * np.array(error_trace) * self.weight_m[i, 0])

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
                # update_new.append(error * -lr)
                # update_new.append(-elig_sum[i] * lr)

                if error > 0:
                    update_new.append(-lr)
                elif error < 0:
                    update_new.append(lr)
                else:
                    update_new.append(0)
            else:
                update_new.append(0)

        update_all = np.expand_dims(np.array(update_new), axis=1) + (self.update_prev * self.momentum)
        self.weight_m = self.weight_m + update_all
        self.weight_m = np.clip(self.weight_m, 0, 1)

        self.update_prev = np.array(update_all)

        return elig
