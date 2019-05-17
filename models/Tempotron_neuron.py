import numpy as np
from collections import deque


class Tempotron:

    def __init__(self, pre_syn, memory, tau_mem=20, omega_rate=0):  # Default 20 and 0.022

            self.pre_syn = pre_syn
            self.memory = memory

            self.tau_mem = tau_mem  # Membrane potential integration time coefficient
            self.tau_syn = tau_mem / 4  # Synaptic potential integration time coefficient

            self.eta = self.tau_mem / self.tau_syn
            self.V_norm = self.eta ** (self.eta / (self.eta - 1)) / (self.eta - 1)

            self.omega_rate = omega_rate  # Weights adjustment. Neuron should spike with fr=5Hz (e.g. N=500->0.022)

            self.theta = 1  # (Initial) Firing threshold
            self.V_rest = 0  # Resting potential

            self.input_memory = np.zeros((pre_syn, memory))
            self.output_memory = np.zeros(memory)
            self.elig = np.zeros((pre_syn, memory))

            if self.omega_rate > 0:
                self.weight_array = np.random.rand(self.pre_syn) * self.omega_rate
            else:
                self.weight_array = np.ones(self.pre_syn)

            self.kernel_mask = self.kernel()[::-1]
            # self.kernel_new = self.kernel_simple()[::-1]  # From Delshad et al.

            self.reset_mask = self.reset()[::-1]

            self.V_t = self.V_rest

            ### Momentum weight update
            self.update_prev = np.zeros(pre_syn)
            self.momentum = 0.4

    def clear(self):
        self.V_t = 0
        self.input_memory = np.zeros((self.pre_syn, self.memory))
        self.output_memory = np.zeros(self.memory)
        self.elig = np.zeros((self.pre_syn, self.memory))

    def run(self, input_frame):

        self.input_memory = np.insert(self.input_memory, 0, input_frame, axis=1)
        self.input_memory = np.roll(self.input_memory, -1, axis=1)
        self.input_memory = np.delete(self.input_memory, 0, 1)

        pre_inputs = (self.input_memory * self.kernel_mask)
        weighted_inputs = (np.rot90(pre_inputs) * self.weight_array)
        resets = (self.output_memory * self.reset_mask)

        V_t = sum(sum(weighted_inputs)) - sum(resets)

        self.elig = np.insert(self.elig, 0, V_t * np.sum(pre_inputs, 1), axis=1)
        self.elig = np.roll(self.elig, -1, axis=1)
        self.elig = np.delete(self.elig, 0, 1)

        self.V_t = V_t

        if V_t >= self.theta:
            spike = 1
        else:
            spike = 0

        self.output_memory = np.insert(self.output_memory, 0, spike)
        self.output_memory = np.roll(self.output_memory, -1)
        self.output_memory = np.delete(self.output_memory, 0)

        return spike

    def kernel(self, memory=0):
        if memory == 0:
            memory = self.memory

        time = np.arange(0, memory, 1)
        kernel = np.zeros(memory)

        for count in range(memory):
            kernel[count] = np.exp(-(time[count]) / self.tau_mem) \
                          - np.exp(-(time[count]) / self.tau_syn)

        kernel = kernel * self.V_norm

        return kernel

    def kernel_simple(self, memory=0):
        if memory == 0:
            memory = self.memory

        time = np.arange(0, memory, 1)
        kernel = np.zeros(memory)

        for count in range(memory):
            kernel[count] = (time[count] / self.tau_mem) * (np.exp(1-(time[count]) / self.tau_mem))

        return kernel

    def reset(self, memory=0):
        if memory == 0:
            memory = self.memory

        time = np.arange(0, memory, 1)
        reset = np.zeros(memory)

        if self.tau_mem >= 1:
            for count in range(memory):
                reset[count] = self.theta * np.exp(-(time[count]) / self.tau_mem)
        else:
            reset[0] = 1

        return reset

    def feedback_weight_update(self, post, pre, error, error_trace, to_update=0.2, lr=0.001):

        ### Select top x% most eligible pre-syn neurons over whole input
        update_partition = np.rint((self.weight_array.__len__()) * to_update).astype(int)

        ### Correlation to post-syn output
        elig_sum = np.sum(self.elig, axis=1)
        most_elig_syn = np.argpartition(elig_sum, -update_partition)[-update_partition:]

        ### Update all blame-wise
        # update_new = []
        # for i in range(0, len(self.weight_m)):
        #     update_new.append(-elig_sum[i] * lr)

        ### Update most_elig_syn blame-wise
        update_new = []
        for i in range(0, len(self.weight_array)):
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

        update_all = np.array(update_new) + (self.update_prev * self.momentum)
        self.weight_array = self.weight_array + update_all
        # self.weight_m = np.clip(self.weight_array, 0, 1)

        self.update_prev = np.array(update_all)

        return elig_sum