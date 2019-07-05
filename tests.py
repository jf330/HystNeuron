import numpy as np
import utils.converter
import utils.training
from autograd import grad
from scipy.sparse import csr_matrix
import os
import csv
import shutil
from tqdm import tqdm

import matplotlib.pyplot as plt
#plt.switch_backend("agg")

from models.hyst_neuron import HystNeuron
from models.hyst_layer import HystLayer

import utils.training as trainer
import utils.plotting as myplt


def aedat_input():
    data = utils.converter.aedat2numpy("/Users/jf330/Downloads/DvsGesture/user01_natural.aedat")
    np.save("/Users/jf330/data_ibm.npy", data)
    # data = np.load("/Users/jf330/data_ibm.npy")

    timepoints = np.rint(data[2, :] * 1000)
    start = timepoints[0].astype(int)
    timepoints = (timepoints - start).astype(int)

    x = data[0, :]
    y = data[1, :]
    pol = data[3, :]

    print("Events: {}".format(x.__len__()))

    # labels = np.zeros(timepoints.__len__())
    labels = np.zeros(timepoints[-1]-timepoints[0])
    with open('/Users/jf330/Downloads/DvsGesture/user01_natural_labels.csv', 'rt')as f:
        data_csv = csv.reader(f)
        i=0
        for row in data_csv:
            if i != 0:
                labels[int(row[1])-start:int(row[2])-start] = int(row[0])
            i += 1

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
    time = np.array(range(0, T))
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


def synt_input(path, datamaker):

    hyst_model = HystNeuron(pre_x=datamaker.n, pre_y=1)


    datamaker.feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()
    hyst_model.weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_output.npy".format(datamaker.n, hyst_model.eta, hyst_model.a))

    datamaker.seed = 0
    datamaker.bg_freq_rate = 0
    data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=True)

    scale = [1,0,0]
    data = np.kron(data, scale)
    hyst_model.a = hyst_model.a/len(scale)
    hyst_model.b = hyst_model.b/len(scale)
    hyst_model.d = hyst_model.d/len(scale)
    hyst_model.g = hyst_model.g/len(scale)

    state = []
    delta_state = []
    reset = []
    i = 0

    while i < data[0].__len__():
        sim = np.where(data[:, i] >= 1)[0]

        print("Clock: {}, inputs: {}".format(i, sim))

        output = hyst_model.decay_step()
        if sim.size != 0:
            hyst_model.event_input(x=sim, y=np.zeros(sim.__len__()).tolist(), values=np.ones_like(sim))

        delta_state.append(output)
        reset.append(hyst_model.reset)
        state.append(hyst_model.state)

        i += 1

    plt.title("Params - h: {}, K: {}, a: {}, b: {}, d:{}, eta: {}".format(hyst_model.h, hyst_model.K, hyst_model.a, hyst_model.b, hyst_model.d, hyst_model.eta))
    plt.ylabel('V(t)')
    plt.xlabel('time')
    plt.ylim((0, 1.3))
    plt.plot(state)
    # plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    # plt.show()

    # plt.plot(delta_state)
    plt.plot(reset)
    plt.axhline(y=hyst_model.K, linestyle="--", color="k")
    plt.legend(loc='best')
    plt.show()


def simple_input():
    hyst_model = HystNeuron(pre_x=1, pre_y=1, omega_rate=-1)
    hyst_model.a = 1
    hyst_model.b = 0.2
    hyst_model.eta = 0.9

    T = 100
    data = np.zeros((1, T))
    data[0, 0] = 1.2

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

    plt.plot(state, label="State")
    # plt.scatter(np.array(range(0, state.__len__())), state, label="State", marker="x")

    plt.plot(reset, label="Reset")
    # plt.scatter(np.array(range(0, reset.__len__())), reset, label="Reset", marker="x")

    # plt.plot(delta_state, label="Output")
    # plt.scatter(np.array(range(0, delta_state.__len__())), delta_state, label="Output", marker="x")

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


def synt_train_many(local_path, datamaker, iter, dt_scale):
    eta_range = np.linspace(0.0, 1.0, iter)
    a_range = np.linspace(0.0, 1.0, iter)

    # for i in tqdm(eta_range):
    for i in eta_range:
        print("Eta: {}".format(i))
        for j in a_range:
            print("A: {}".format(j))
            synt_train(local_path, datamaker, dt_scale, eta=i, a=j)

            
def sigmoid(x):
    sig = []
    for x_t in x:
        sig.append((1 / (1 + np.exp(-(x_t - 1) * 250))))
    return sig


# sig(x) works for numpy arrays and values
def sig(x):
    return 1 / (1 + np.exp(-200 * (x - 1)))


# TODO: Rewrite my state array so that it contains sig(state) so that sig_prime returns x - x^2
def sig_prime(x):
    return sig(x) * (1 - sig(x))


def synt_network(path, datamaker, eta=-1, a=-1):
    lr = 0.00001
    # to_update = 0.2  # Do I want to train most eligible?
    epochs = 1000
    omega_rate_layer = 0.8
    omega_rate_out = 0.8

    noise = True
    datamaker.bg_freq_rate = 0

    plotting = True
    readout = "output"

    no_layer_n = 2

    if eta < 0 and a < 0:
        layer = HystLayer(omega_rate=omega_rate_layer, each_pre_x=datamaker.n, each_pre_y=1, n=no_layer_n)
        out = HystNeuron(omega_rate=omega_rate_out, pre_x=no_layer_n, pre_y=1)
    else:
        layer = HystLayer(omega_rate=omega_rate_layer, each_pre_x=datamaker.n, each_pre_y=1, n=no_layer_n, switch=eta, a=a)
        out = HystNeuron(omega_rate=omega_rate_out, pre_x=no_layer_n, pre_y=1, eta=eta, a=a)

    error = []
    for e in range(0, epochs):
        # if e % 1000 == 0:
        print(e)
        out_state = []
        out_out = []
        out_reset = []
        neuron_states = [x[:] for x in [[]] * no_layer_n]
        neuron_outs = [x[:] for x in [[]] * no_layer_n]
        neuron_resets = [x[:] for x in [[]] * no_layer_n]
        out.clear()

        data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise)

        neuron_states_actual = []

        for bin in range(0, len(data[1])):

            simul_events = np.where(data[:, bin] >= 1)[0]
            # print("Clock: {}, inputs: {}".format(bin, simul_events))

            if simul_events.size != 0:
                layer.event_input(x=simul_events, y=np.zeros(simul_events.__len__()).tolist(),
                                     values=np.ones_like(simul_events))
            out_A = layer.decay_step()
            spikes = []
            neurons = layer.neurons
            rand_no = rand.random()
            for i in range(0, len(neurons)):
                sig_val = sig(neurons[i].state) > rand_no
                for j in range(0, len(neurons)):
                    if j != i:
                        neurons[j].reset += sig_val

                spikes.append(sig_val)
                neuron_outs[i].append(sig_val)
                neuron_states[i].append(neurons[i].state)  # For graphing purposes

            state_array = np.array([neuron_states[0][bin], neuron_states[1][bin]])  # TODO: Generalise this
            neuron_states_actual.append(np.reshape(state_array, (2, 1)))

            events_O = np.where(np.array(spikes) >= 1)[0]
            if events_O.size != 0:
                out.event_input(x=events_O, y=np.zeros(events_O.__len__()).tolist(), values=np.ones_like(events_O))
            out_O = out.decay_step()  # May not need the value?
            out_spike = sig(out.state) > rand_no

            out_state.append(out.state)
            out_reset.append(out.reset)
            out_out.append(out_spike)

        # Error calculations
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

        # Calculate error trace TODO: Test out how to punish spikes outside of features
        error_trace_O = trainer.calc_error(out.K, desired_state, out_out)
        error_out = sum(error_trace_O)
        error.append(error_out)

        # Back-propagation to work out weight changes TODO: Batch train
        del_k = []
        del_j = []
        for i in range(0, len(error_trace_O)):
            del_k.append(-np.array(error_trace_O[i]))

        dw_jk = np.zeros_like(out.weight_m)
        for i in range(0, len(del_k)):
            dw_jk = dw_jk + np.dot(sig(neuron_states_actual[i]), del_k[i].T)
            del_j.append(del_k[i] * out.weight_m * sig_prime(neuron_states_actual[i]))
        dw_jk *= - lr

        dw_ij = np.zeros_like(layer.neurons[0].weight_m)
        for i in range(0, len(del_j)):
            data_step = np.expand_dims(np.array(data.T[i]), 1)
            dw_ij = dw_ij + np.dot(sig(data_step), del_j[i].T)
        dw_ij *= - lr

        # Make changes to weights
        for i in range(0, len(layer.neurons)):
            layer.neurons[i].update_weight(dw_ij[i])
        out.update_weight(dw_jk)

    # Plot data
    fig, axes = plt.subplots(3, 0, sharex=True, sharey=True)
    n1 = fig.add_subplot(212)
    n1.plot(out_out)
    fig.text(0.06, 0.25, 'Out', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.03, 'Time', ha='center', va='center')
    # n1.xlabel('Time')
    # n1.ylabel('Out')
    n1.set_title('Output neuron')
    n2 = fig.add_subplot(321)
    n2.plot(neuron_outs[0])
    fig.text(0.06, 0.75, 'Out', ha='center', va='center', rotation='vertical')
    fig.text(0.3, 0.57, 'Time', ha='center', va='center')
    # n2.xlabel('Time')
    # n2.ylabel('Out')
    n2.set_title('Hidden neuron 1')
    n3 = fig.add_subplot(322)
    n3.plot(neuron_outs[1])
    fig.text(0.5, 0.75, 'Out', ha='center', va='center', rotation='vertical')
    fig.text(0.7, 0.57, 'Time', ha='center', va='center')
    # n3.xlabel('Time')
    # n3.ylabel('Out')
    n3.set_title('Hidden neuron 2')
    plt.show()

    plt.plot(out_state, label="A state")
    plt.plot(out_reset, label="A reset")
    plt.plot(out_out, label="A out")
    plt.axhline(y=out.K, linestyle="--", color="k")

    markers = datamaker.add_marker(time_occur, fea_order, 1, 0.05)
    for marker in markers:
        plt.gca().add_patch(marker)
        plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
        plt.gca().add_patch(marker)

    plt.ylabel('Neural dynamics')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()

    plt.plot(error, label="error")
    plt.show()
    

def synt_train(path, datamaker, dt_scale, eta=0.9, a=0.2):
    ### Training setup
    lr = 0.0001
    to_update = 0.2
    epochs = 25000
    omega_rate = 0.3

    noise = True
    datamaker.bg_freq_rate = 1

    plotting = False
    readout = "output"

    if eta < 0 and a < 0:
        neuron_A = HystNeuron(omega_rate=omega_rate, pre_x=datamaker.n, pre_y=1)
    else:
        neuron_A = HystNeuron(omega_rate=omega_rate, pre_x=datamaker.n, pre_y=1, eta=eta, a=a)

    # neuron_A.a = neuron_A.a*dt_scale
    # neuron_A.b = neuron_A.b*dt_scale
    # neuron_A.d = neuron_A.d*dt_scale
    # neuron_A.g = neuron_A.g*dt_scale

    ### Load pre-generated features
    features_path = path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)
    if os.path.isfile(features_path):
        datamaker.feature_list = np.load(features_path, allow_pickle=True).item()
    else:
        np.save(features_path, datamaker.feature_list, allow_pickle=True)

    ### Load pre-trained weights
    # neuron_A.weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_{}_Epoch_{}.npy".format(datamaker.n, neuron_A.eta, neuron_A.a, readout, 9999))

    neuron_A_error = []
    for e in range(0, epochs):
        if e % 100 == 0:
            print("Epoch {}".format(e))
        #     np.save(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_{}_Epoch_{}.npy".format(neuron_A.pre_syn, neuron_A.eta,np.around(neuron_A.a,decimals=3), readout, e), neuron_A.weight_m)

        ### Init. history arrays
        neuron_A_state = []
        neuron_A_out = []
        neuron_A_reset = []
        neuron_A.clear()

        data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise, fea_mode=3)

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
            elif fea_order[count] == 2:
                desired_state[index[count]:index[count] + T_fea_local] = 3
            # elif fea_order[count] == 3:
            #     desired_state[index[count]:index[count] + T_fea_local] = 5
            # elif fea_order[count] == 4:
            #     desired_state[index[count]:index[count] + T_fea_local] = 5
            # elif fea_order[count] == 5:
            #     desired_state[index[count]:index[count] + T_fea_local] = 6

            index += np.rint(T_fea_local).astype(int)
            count += 1

        # desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2
        desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2 + n_fea_occur[2] * 3
        # desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2 + n_fea_occur[2] * 3 + n_fea_occur[3] * 4
        # desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2 + n_fea_occur[2] * 3 + n_fea_occur[3] * 4 + n_fea_occur[4] * 5

        if readout == "state":
            error_trace = trainer.calc_error(neuron_A.K, desired_state, neuron_A_state)
            error = np.where(np.array(neuron_A_state) >= neuron_A.K)[0].__len__() - desired_spikes

            synt_reset = trainer.calc_synt_reset(data, neuron_A.b)
            neuron_A.feedback_weight_update(neuron_A_state, synt_reset, error, error_trace, to_update=to_update, lr=lr)
            # neuron_A.feedback_weight_update(neuron_A_state, np.rot90(data), error, error_trace, to_update=to_update, lr=lr)

        elif readout == "output":
            error_trace = trainer.calc_error(neuron_A.K, desired_state, neuron_A_out)
            error = np.where(np.array(sig_output) >= neuron_A.K)[0].__len__() - desired_spikes
            # error = sum(error_trace)

            synt_reset = trainer.calc_synt_reset(data, neuron_A.b)
            neuron_A.feedback_weight_update(neuron_A_out, synt_reset, error, error_trace, to_update=to_update, lr=lr)
            # neuron_A.feedback_weight_update(neuron_A_out, np.rot90(data), error, error_trace, to_update=to_update, lr=lr)

        elif readout == "state-probab":
            random_arr = np.random.rand((len(neuron_A_out)))
            sig_state = np.array(sigmoid(neuron_A_state))
            spike_idxs = np.where(random_arr < sig_state)[0]
            sig_output = np.zeros_like(neuron_A_state)
            sig_output[spike_idxs] = 1

            error_trace = trainer.calc_error(neuron_A.K, desired_state, sig_output)
            error = np.where(np.array(sig_output) >= neuron_A.K)[0].__len__() - desired_spikes
            # error = sum(error_trace)

            synt_reset = trainer.calc_synt_reset(data, neuron_A.b)
            neuron_A.feedback_weight_update(neuron_A_state, synt_reset, error, error_trace, to_update=to_update, lr=lr)
            # neuron_A.feedback_weight_update(neuron_A_out, np.rot90(data), error, error_trace, to_update=to_update, lr=lr)

        elif readout == "reset":
            error_trace = trainer.calc_error(neuron_A.K, desired_state, neuron_A_reset)
            error = np.where(np.array(neuron_A_reset) >= neuron_A.K)[0].__len__() - desired_spikes

            synt_reset = trainer.calc_synt_reset(data, neuron_A.b)
            neuron_A.feedback_weight_update(neuron_A_reset, synt_reset, error, error_trace, to_update=to_update, lr=lr)
            # neuron_A.feedback_weight_update(neuron_A_reset, np.rot90(data), error, error_trace, to_update=to_update, lr=lr)

        # print("Error: {}".format(error))
        neuron_A_error.append(error)

    if plotting:
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
        plt.plot(neuron_A_reset, label="A reset")
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

        ### Plot error
        plt.scatter(np.array(range(0, neuron_A_error.__len__())), neuron_A_error, marker="x", label="Error (A)")
        # plt.plot(neuron_A_error, label="Error (A)")
        plt.ylabel('Error')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()

        ### Plot trial error
        markers = datamaker.add_marker(time_occur, fea_order, 1, 0.05)
        for marker in markers:
            plt.gca().add_patch(marker)
            plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
            plt.gca().add_patch(marker)
        plt.plot(error_trace)
        plt.ylabel('Error')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()

    ### Print spike times
    print(np.where(np.array(neuron_A_state) >= 1)[0])

    ### Save trained weights
    np.save(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_{}_Epoch_{}.npy".format(neuron_A.pre_syn, neuron_A.eta, np.around(neuron_A.a, decimals=3), readout, e), neuron_A.weight_m)


def aedat_train(path, datamaker, eta=-1, a=-1):
    # Training setup
    lr = 0.001
    to_update = 0.1
    epochs = 1
    omega_rate = 0.08
    noise = True
    cwd = os.path.dirname(__file__)

    # data = utils.converter.aedat2numpy("/Users/jf330/Downloads/DvsGesture/user01_natural.aedat")
    # np.save("/Users/jf330/data_ibm.npy", data)
    data = np.load("/Users/jf330/data_ibm.npy")

    timepoints = np.rint(data[2, :] * 1000)
    start = timepoints[0].astype(int)
    timepoints = (timepoints - start).astype(int)

    x = data[0, :]
    y = data[1, :]
    pol = data[3, :]

    print("Events: {}".format(x.__len__()))

    labels = np.zeros(timepoints[-1] - timepoints[0])
    with open('/Users/jf330/Downloads/DvsGesture/user01_natural_labels.csv', 'rt')as f:
        data_csv = csv.reader(f)
        i = 0
        for row in data_csv:
            if i != 0:
                labels[int(row[1]) - start:int(row[2]) - start] = int(row[0])
            i += 1

    # datamaker.feature_list = np.load(cwd + "/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()
    np.save(cwd + "/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea), datamaker.feature_list)

    if eta < 0 and a < 0:
        neuron_A = HystNeuron(omega_rate=omega_rate, pre_x=240, pre_y=128)
    else:
        neuron_A = HystNeuron(omega_rate=omega_rate, pre_x=240, pre_y=128, eta=eta, a=a)

    neuron_A_error = []
    for e in range(0, epochs):
        print("Epoch {}".format(e))
        # History arrays
        neuron_A_state = []
        neuron_A_out = []
        neuron_A_reset = []
        neuron_A.clear()

        # T = timepoints[-1]
        T = 1000
        i = 0
        while i < T:
            sim = np.where(timepoints == i)[0]
            print("Clock: {}, inputs: {}".format(i, sim.__len__()))

            if sim.size != 0:
                neuron_A.event_input(x=x[sim], y=y[sim], values=np.ones_like(x[sim]))
                # neuron_A.event_input(x=x[sim], y=y[sim], values=pol[sim])

            out_A = neuron_A.decay_step()

            neuron_A_state.append(neuron_A.state)
            neuron_A_reset.append(neuron_A.reset)
            neuron_A_out.append(out_A)
            i+=1

        error_trace = trainer.calc_error(neuron_A.K, labels[0:T], neuron_A_state)
        error = sum(error_trace)

        neuron_A.events_weight_update(neuron_A_state, x, y, error, error_trace, to_update=to_update, lr=lr)

        # print("Error: {}".format(error))
        neuron_A_error.append(error)


    ### Plot input data raster
    plt.plot(labels)
    plt.show()

    ### Plot reaction
    plt.axhline(y=neuron_A.K, linestyle="--", color="k")
    plt.plot(neuron_A_out, label="A out")
    plt.ylabel('O(t)')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()

    plt.plot(neuron_A_state, label="A state")
    plt.plot(neuron_A_reset, label="A reset")
    plt.plot(neuron_A_out, label="A out")
    plt.axhline(y=neuron_A.K, linestyle="--", color="k")
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

    print(np.where(np.array(neuron_A_state) >= 1)[0])

    ### Save trained weights
    # np.save(cwd + "/weights_N_{}_Eta_{}_A_{}_noisy".format(neuron_A.pre_syn, neuron_A.eta, neuron_A.a), neuron_A.weight_m)


def synt_train_bp(path, datamaker):
    # Training setup
    lr = 0.0002
    to_update = 0.1
    epochs = 10000
    omega_rate = 0.02

    noise = True
    datamaker.bg_freq_rate = 1

    plotting = True
    readout = "output"

    # datamaker.feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()
    np.save(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea), datamaker.feature_list)

    neuron_layer_H = HystLayer(omega_rate=omega_rate, pre_x=datamaker.n, pre_y=1, n=10)
    neuron_O = HystNeuron(omega_rate=omega_rate, pre_x=10, pre_y=1)

    neuron_O_error = []
    for e in range(0, epochs):
        print("Epoch {}".format(e))
        # History arrays
        neuron_O_state = []
        neuron_O_out = []
        neuron_O_reset = []
        neuron_O.clear()

        data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise)

        ### Present stimulus
        for bin in range(0, len(data[1])):

            simul_events = np.where(data[:, bin] >= 1)[0]
            # print("Clock: {}, inputs: {}".format(bin, simul_events))

            if simul_events.size != 0:
                neuron_layer_H.event_input(x=simul_events, y=np.zeros(simul_events.__len__()).tolist(), values=np.ones_like(simul_events))
            out_H = neuron_layer_H.decay_step()

            ### SHOULD IT BE A CONSTANT INPUT FLOW TO THE OUTPUT LAYER
            # simul_events_H = np.where(out_H >= 0)
            # neuron_O.event_input(x=simul_events_H, y=np.zeros(simul_events_H.__len__()).tolist(), values=np.ones_like(simul_events_H))

            ### OR ONLY WHEN "SPIKING"
            simul_events_H = np.where(out_H >= 1)[0]
            if simul_events.size != 0:
                neuron_O.event_input(x=simul_events_H, y=np.zeros(simul_events_H.__len__()).tolist(), values=np.ones_like(simul_events_H))

            out_O = neuron_O.decay_step()

            neuron_O_state.append(neuron_O.state)
            neuron_O_reset.append(neuron_O.reset)
            neuron_O_out.append(out_O)

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

        if readout == "state":
            error_trace = trainer.calc_error(neuron_O.K, desired_state, neuron_O_state)
            error = np.where(np.array(neuron_O_state) >= neuron_O.K)[0].__len__() - desired_spikes


        elif readout == "output":
            error_trace = trainer.calc_error(neuron_O.K, desired_state, neuron_O_out)
            error = np.where(np.array(neuron_O_out) >= neuron_O.K)[0].__len__() - desired_spikes

        # for neuron in neuron_layer_H.neurons:


        # print("Error: {}".format(error))
        neuron_O_error.append(error)

    if plotting:
        ### Plot input data raster
        markers = datamaker.add_marker(time_occur, fea_order, neuron_O.pre_syn - 1, neuron_O.pre_syn / 40)
        myplt.plot_features(markers, data)

        ### Plot reaction
        plt.axhline(y=neuron_O.K, linestyle="--", color="k")

        markers = datamaker.add_marker(time_occur, fea_order, 1, 0.05)
        for marker in markers:
            plt.gca().add_patch(marker)
            plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
            plt.gca().add_patch(marker)
        plt.plot(neuron_O_out, label="A out")
        plt.ylabel('O(t)')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()

        plt.plot(neuron_O_state, label="A state")
        plt.plot(neuron_O_reset, label="A reset")
        plt.plot(neuron_O_out, label="A out")
        plt.axhline(y=neuron_O.K, linestyle="--", color="k")

        markers = datamaker.add_marker(time_occur, fea_order, 1, 0.05)
        for marker in markers:
            plt.gca().add_patch(marker)
            plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
            plt.gca().add_patch(marker)

        plt.ylabel('Neural dynamics')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()

        ### Plot error
        plt.scatter(np.array(range(0, neuron_O_error.__len__())), neuron_O_error, marker="x", label="Error (A)")
        # plt.plot(neuron_A_error, label="Error (A)")
        plt.ylabel('Error')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()

        ### Plot trial error
        # markers = datamaker.add_marker(time_occur, fea_order, 1, 0.05)
        # for marker in markers:
        #     plt.gca().add_patch(marker)
        #     plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
        #     plt.gca().add_patch(marker)
        # plt.plot(error_trace_O)
        # plt.ylabel('Error')
        # plt.xlabel('Time')
        # plt.legend(loc='best')
        # plt.show()

    print(np.where(np.array(neuron_O_state) >= 1)[0])

    ### Save trained weights
    # np.save(cwd + "/weights_N_{}_Eta_{}_A_{}_noisy".format(neuron_A.pre_syn, neuron_A.eta, neuron_A.a), neuron_A.weight_m)
