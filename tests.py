import numpy as np
import utils.converter
import utils.training
import autograd.numpy as npa
from autograd import grad
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os

from models.hyst_neuron import HystNeuron
from models.hyst_layer import HystLayer

import utils.training as trainer
import utils.plotting as myplt


def aedat_input():
    # data = utils.converter.aedat2numpy("/Users/jf330/Desktop/Gesture/Bruno2_dvs.aedat")
    # np.save("/Users/jf330/data_bruno.npy", data)
    data = np.load("/Users/jf330/data.npy")

    timepoints = np.rint(data[2, :] * 1000)
    timepoints = (timepoints - timepoints[0]).astype(int)

    x = data[0, :]
    y = data[1, :]
    pol = data[3, :]

    print("Events: {}".format(x.__len__()))

    hyst_model = HystNeuron(pre_x=240, pre_y=128)

    T = 1500
    # T = timepoints[-1]
    print("T (whole): {}".format(timepoints[-1]))
    state = []
    delta_state = []

    i = 0
    while i < T:
        sim = np.where(timepoints == i)[0]
        print("Clock: {}, inputs: {}".format(i, sim.__len__()))

        if sim.size != 0:
            hyst_model.event_input(x=x[sim], y=y[sim], values=np.ones_like(x[sim]))
            # hyst_model.event_input(x=x[sim], y=y[sim], values=pol[sim])
        output = hyst_model.decay_step()
        delta_state.append(output)

        state.append(hyst_model.state)

        i += 1

    plt.plot(state)
    plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    plt.show()

    plt.plot(delta_state)
    plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    plt.show()


def cont_current_input():
    hyst_model = HystNeuron(pre_x=1, pre_y=1)

    T = 200
    current = 0.19
    time = np.array(range(0,T))
    state = []
    delta_state = []
    i = 0
    while i < T:
        print("Clock: {}, inputs: {}".format(i,current))

        hyst_model.cont_current(current)
        output = hyst_model.decay_step()
        delta_state.append(output)

        state.append(hyst_model.state)

        i += 1

    # plt.title("Params - h: {}, K: {}, a: {}, b: {}, eta: {}".format(hyst_model.h, hyst_model.K, hyst_model.a, hyst_model.b, hyst_model.eta))
    # plt.scatter(time, state)
    # plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    # plt.show()

    plt.title("Params - h: {}, K: {}, a: {}, b: {}, eta: {}".format(hyst_model.h, hyst_model.K, hyst_model.a, hyst_model.b, hyst_model.eta))
    plt.plot(state)
    plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    plt.show()

    plt.plot(delta_state)
    plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    plt.show()


def synt_input(datamaker):
    data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=True)

    hyst_model = HystNeuron(pre_x=datamaker.n, pre_y=1)

    time = np.array(range(0, data[0].__len__()))
    state = []
    delta_state = []
    i = 0
    while i < data[0].__len__():
        sim = np.where(data[:,i] >= 1)[0]
        print("Clock: {}, inputs: {}".format(i,sim))

        if sim.size != 0:
            hyst_model.event_input(x=sim, y=np.zeros(sim.__len__()).tolist(), values=np.ones_like(sim))
        output = hyst_model.decay_step()
        delta_state.append(output)

        state.append(hyst_model.state)

        i += 1

    plt.plot(state)
    plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    plt.show()

    plt.plot(delta_state)
    plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    plt.show()


def simple_input():
    hyst_model = HystNeuron(pre_x=1, pre_y=1, omega_rate=-1)
    T = 40
    data = np.zeros((1,T))
    data[0,10] = 1.1

    delta_state = []
    state = []
    reset = []

    i = 0
    while i < data[0].__len__():

        sim = np.where(data[:,i] >= 1)[0]
        print("Clock: {}, inputs: {}".format(i,sim))

        if sim.size != 0:
            hyst_model.event_input(x=sim, y=np.zeros(sim.__len__()).tolist(), values=data[:,i])
        output = hyst_model.decay_step()

        delta_state.append(output)
        reset.append(hyst_model.reset)
        state.append(hyst_model.state)
        i += 1

    plt.axhline(y=hyst_model.K, linestyle="--", color="k")

    # plt.plot(state, label="State")
    plt.scatter(np.array(range(0, state.__len__())), state, label="State")

    # plt.plot(reset, label="Reset")
    plt.scatter(np.array(range(0, reset.__len__())), reset, label="Reset")

    # plt.plot(delta_state, label="Output")
    plt.scatter(np.array(range(0, delta_state.__len__())), delta_state, label="Output")

    plt.legend(loc='best')
    plt.show()


def simple_STDP_train():
    pre_1 = HystNeuron(pre_x=1, pre_y=1, omega_rate=-1)
    pre_2 = HystNeuron(pre_x=1, pre_y=1, omega_rate=-1)
    post = HystNeuron(pre_x=1, pre_y=1, omega_rate=-1)

    epochs = 10
    T = 40
    data = np.zeros((2, T))
    data[0, 10] = 1.1

    for e in range(0, epochs):

        post_delta_state = []
        post_state = []
        post_reset = []

        i = 0
        while i < data[0].__len__():

            sim = np.where(data[:,i] >= 1)[0]

            print("Clock: {}, inputs: {}".format(i,sim))

            if sim.size != 0:
                post.event_input(x=sim, y=np.zeros(sim.__len__()).tolist(), values=data[:,sim])
            output = post.decay_step()

            post_delta_state.append(output)
            post_reset.append(post.reset)
            post_state.append(post.state)
            i += 1

    plt.axhline(y=post.K, linestyle="--", color="k")

    # plt.plot(state, label="State")
    plt.scatter(np.array(range(0, post_state.__len__())), post_state, label="State")

    # plt.plot(reset, label="Reset")
    plt.scatter(np.array(range(0, post_reset.__len__())), post_reset, label="Reset")

    # plt.plot(delta_state, label="Output")
    plt.scatter(np.array(range(0, post_delta_state.__len__())), post_delta_state, label="Output")

    plt.legend(loc='best')
    plt.show()


def synt_train_many(datamaker, iter):

    params = np.linspace(0, 1, iter)

    for i in params:  # FIXME do tqdm progress bar
        print("Eta: {}".format(i))
        for j in params:
            synt_input_train(datamaker, eta=i, a=j)


def synt_input_train(datamaker, eta=-1, a=-1):
    # Training setup
    lr = 0.01
    to_update = 0.4
    epochs = 2000
    omega_rate = 0.15
    noise = False
    cwd = os.path.dirname(__file__)

    # datamaker.feature_list = np.load(cwd + "feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()
    np.save(cwd + "/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea), datamaker.feature_list)

    if eta < 0 and a < 0:
        neuron_A = HystNeuron(omega_rate=omega_rate, pre_x=datamaker.n, pre_y=1)
    else:
        neuron_A = HystNeuron(omega_rate=omega_rate, pre_x=datamaker.n, pre_y=1, eta=eta, a=a)

    neuron_A_error = []
    for e in range(0, epochs):
        print("Epoch {}".format(e))
        # History arrays
        neuron_A_state = []
        neuron_A_out = []
        neuron_A_reset = []
        neuron_A.clear()

        data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise)

        ### Present stimulus
        for bin in range(0, len(data[1])):

            simul_events = np.where(data[:, bin] >= 1)[0]
            # print("Clock: {}, inputs: {}".format(bin, simul_events))

            if simul_events.size != 0:
                neuron_A.event_input(x=simul_events, y=np.zeros(simul_events.__len__()).tolist(), values=np.ones_like(simul_events))
            out_A = neuron_A.decay_step()

            neuron_A_state.append(neuron_A.state)
            neuron_A_reset.append(neuron_A.reset)
            neuron_A_out.append(out_A)

        ### Weight update
        desired_state = np.zeros(len(data[1]))
        count = 0
        index = np.rint(time_occur / datamaker.dt).astype(int)
        while count < len(index):
            fea = datamaker.feature_list['feature_' + str(fea_order[count])]
            T_fea_local = fea[0].size

            if fea_order[count] == 0:
                desired_state[index[count]:index[count] + T_fea_local] = 1
            elif fea_order[count] == 1:
                desired_state[index[count]:index[count] + T_fea_local] = 2

            index += np.rint(T_fea_local).astype(int)
            count += 1

        desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2

        error, error_trace = trainer.calc_integrals(neuron_A.K, desired_state, neuron_A_state, fea_order, time_occur, datamaker)  # Also seems to work
        neuron_A.feedback_weight_update(neuron_A_state, np.rot90(data), error, error_trace, to_update=to_update, lr=lr)

        # print("Error: {}".format(error))
        neuron_A_error.append(error)


    ### Plot input data raster
    markers = datamaker.add_marker(time_occur, fea_order, neuron_A.pre_syn - 1, neuron_A.pre_syn / 40)
    myplt.plot_features(markers, data)

    ### Plot reaction
    plt.axhline(y=neuron_A.K, linestyle="--", color="k")

    markers = datamaker.add_marker(time_occur, fea_order, 1, 0.05)
    for marker in markers:
        plt.gca().add_patch(marker)
        plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
        plt.gca().add_patch(marker)
    plt.plot(neuron_A_out, label="A out")
    plt.ylabel('O(t)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()

    plt.plot(neuron_A_state, label="A state")
    plt.plot(neuron_A_reset, label="A hyst")
    plt.plot(neuron_A_out, label="A out")
    plt.axhline(y=neuron_A.K, linestyle="--", color="k")

    markers = datamaker.add_marker(time_occur, fea_order, 1, 0.05)
    for marker in markers:
        plt.gca().add_patch(marker)
        plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
        plt.gca().add_patch(marker)

    plt.ylabel('Neural dynamics')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()

    # Plot error
    plt.scatter(np.array(range(0, neuron_A_error.__len__())), neuron_A_error, marker="x", label="Error (A)")
    # plt.plot(neuron_A_error, label="Error (A)")
    plt.ylabel('Error')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()

    ### Save trained weights
    np.save(neuron_A.weight_m, "weights_N_{}_Eta_{}_A_{}".format(neuron_A.pre_syn, neuron_A.eta, neuron_A.a))
