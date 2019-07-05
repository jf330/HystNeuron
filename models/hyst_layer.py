from models.hyst_neuron import HystNeuron
import numpy as np


class HystLayer:
    def __init__(self, h=150, K=1, eta=0.9, a=0.2, b=0.5, omega_rate=0.5, each_pre_x=1, each_pre_y=1, n=1):
        self.n = n
        self.neurons = []
        for count in range(0, self.n):
            buffer = HystNeuron(h, K, eta, a, b, omega_rate=omega_rate, pre_x=each_pre_x, pre_y=each_pre_y)
            self.neurons.append(buffer)

        self.history_state = []
        self.history_reset = []
        self.history_out = []
        self.frame_num = 0

    def event_input(self, x, y, values):
        for idx in range(0, len(self.neurons)):
            self.neurons[idx].event_input(x, y, values)

    def decay_step(self):
        output = []

        for idx in range(0, len(self.neurons)):
            out = self.neurons[idx].decay_step()

            output.append(out)
            if self.frame_num == 0:
                self.history_state.append([self.neurons[idx].state])
                self.history_reset.append([self.neurons[idx].reset])
                self.history_out.append([self.neurons[idx].out])
            else:
                self.history_state[idx].append(self.neurons[idx].state)
                self.history_reset[idx].append(self.neurons[idx].reset)
                self.history_out[idx].append([self.neurons[idx].out])

        self.frame_num += 1

        return output

    def clear(self):
        self.history_state = []
        self.history_reset = []
        self.history_out = []
        for neuron in self.neurons:
            neuron.clear()
            self.frame_num = 0

    def lateral_inhib(self):
        print("TODO")
