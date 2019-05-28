import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO
import os

from models.Tempotron_neuron import Tempotron

import utils.training as trainer
import utils.plotting as myplt


def gekko_ode_input(path, datamaker):
    cwd = os.path.dirname(__file__)
    noise = False
    datamaker.feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()

    ### Create GEKKO model
    m = GEKKO()
    h = 2000
    K = 1
    eta = 0.9
    # a = 0.2
    # b = 0.5
    # d = 1
    a = 0.2
    b = 0.5
    d = 1

    weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_output.npy".format(datamaker.n, eta, a))

    # a = a/2
    # b = b/2
    # d = d/2

    ### Spatio-temporal inputs
    datamaker.seed = 0
    data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise)

    # input_data = np.sum(weight_m * data, axis=0)
    # print(input_data[35:45])
    # input_data[39] = 0.8
    # time = input_data.__len__()
    time = 100
    # time = 100*2
    input_data = np.zeros(time)
    input_data[1] = 1.3
    m.time = np.linspace(0, time-1, time)  # time points
    u = m.Param(value=input_data)

    ### ODE system
    V = m.Var(0.0)
    X = m.Var(0.0)

    m.Equation(
        V.dt() == -((eta * X * V) + ((1-eta) * a * V)) + u
        # V.dt() == -((eta * X/2 * V) + ((1-eta) * a * V)) + u
    )

    m.Equation(
        X.dt() == ((V ** h) / (K ** h + V ** h)) * d - b * X
        # X.dt() == (np.power(V, h) / (np.power(K, h) + np.power(V, h))) - b * X
        # X.dt() == (0.5 * (1 + np.tanh(h * (V - K)))) - b * X
    )

    plt.title("Params - h: {}, K: {}, a: {}, b: {}, d: {}, eta: {}".format(h, K, a, b, d, eta))

    ### Solve ODE
    m.options.IMODE = 4
    m.solve()

    print("Max X: {}, Max V: {}".format(max(X), max(V)))

    ### Plot results
    plt.axhline(y=K, linestyle="--", color="k")
    plt.plot(m.time, X, 'b-', label='H(t)')
    plt.plot(m.time, V, 'r-', label='V(t)')

    plt.ylim((0, 1.3))
    plt.ylabel('V(t)')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()


def gekko_ode_train(path, datamaker, eta=-1, a=-1):
    ### Training setup
    lr = 0.0001
    to_update = 0.2
    epochs = 1
    omega_rate = 0.8

    noise = False
    datamaker.bg_freq_rate = 0

    plotting = False
    readout = "output"

    ### Load pre-generated features
    features_path = path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)
    if os.path.isfile(features_path):
        datamaker.feature_list = np.load(features_path, allow_pickle=True).item()
    else:
        np.save(features_path, datamaker.feature_list, allow_pickle=True)

    ### Load pre-trained weights
    # weight_m = np.load(path + /"weights/weights_N_{}_Eta_{}_A_{}.npy".format(datamaker.n, neuron_A.eta, neuron_A.a))
    weight_m = np.random.rand(datamaker.n, 1) * omega_rate

    neuron_A_error = []
    for e in range(0, epochs):
        if e % 1000 == 0:
            print("Epoch {}".format(e))
            # datamaker.bg_freq_rate += 0.1

        ### Init. history arrays
        neuron_A_state = []
        neuron_A_out = []
        neuron_A_reset = []

        data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise, fea_mode=3)
        input_data = np.sum(weight_m * data, axis=0)

        ### Create GEKKO model
        m = GEKKO()
        h = 2000
        K = 1
        eta = 0.9
        a = 0.2
        b = 0.5
        d = 1

        m.time = np.linspace(0, input_data.__len__() - 1, input_data.__len__())  # time points
        u = m.Param(value=input_data)

        ### ODE system
        V = m.Var(0.0)
        R = m.Var(0.0)

        m.Equation(V.dt() == -((eta * R * V) + ((1 - eta) * a * V)) + u)
        m.Equation(R.dt() == (V ** h / (K ** h + V ** h)) * d - b * R)

        ### Solve ODE
        m.options.IMODE = 7
        m.solve()

        neuron_A_state.append(V)
        neuron_A_reset.append(R)
        neuron_A_out.append(V)

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
            # elif fea_order[count] == 2:
            #     desired_state[index[count]:index[count] + T_fea_local] = 3
            # elif fea_order[count] == 3:
            #     desired_state[index[count]:index[count] + T_fea_local] = 4
            # elif fea_order[count] == 4:
            #     desired_state[index[count]:index[count] + T_fea_local] = 5
            # elif fea_order[count] == 5:
            #     desired_state[index[count]:index[count] + T_fea_local] = 6

            index += np.rint(T_fea_local).astype(int)
            count += 1

        desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2

        if readout == "state":
            # error_trace = trainer.calc_error(K, desired_state, neuron_A_state)
            error = np.where(np.array(neuron_A_state) >= K)[0].__len__() - desired_spikes

            synt_reset = trainer.calc_synt_reset(data, b)
            weight_m = feedback_weight_update(neuron_A_state, synt_reset, error, to_update=to_update, lr=lr)

        elif readout == "output":
            # error_trace = trainer.calc_error(K, desired_state, neuron_A_out)
            error = np.where(np.array(neuron_A_out) >= K)[0].__len__() - desired_spikes

            synt_reset = trainer.calc_synt_reset(data, b)
            weight_m = feedback_weight_update(weight_m, neuron_A_out, synt_reset, error, to_update=to_update, lr=lr)

        # print("Error: {}".format(error))
        neuron_A_error.append(error)

    plt.title("Params - h: {}, K: {}, a: {}, b: {}, d: {}, eta: {}".format(h, K, a, b, d, eta))

    ### Plot results
    plt.axhline(y=K, linestyle="--", color="k")
    plt.plot(m.time, R, 'b-', label='H(t)')
    plt.plot(m.time, V, 'r-', label='V(t)')

    plt.ylim((0, 1.3))
    plt.ylabel('V(t)')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()

def feedback_weight_update(weight_m, post, pre, error, to_update=0.2, lr=0.001):

    ### Select top x% most eligible pre-syn neurons over whole input
    update_partition = np.rint((weight_m.__len__()) * to_update).astype(int)

    ### Correlation to post-syn output
    elig = []
    for i in range(0, len(weight_m)):
        elig.append(np.array(pre[:, i]) * np.array(post))
        # elig.append(np.array(pre[:, i]) * np.array(abs(error_trace)))
        # elig.append(np.array(pre[:, i]) * np.array(error_trace))
        # elig.append(np.array(pre[:, i]) * np.array(post) * np.array(error_trace))
        # elig.append(np.array(pre[:, i]) * np.array(post) * np.array(error_trace) * self.weight_m[i, 0])

    elig_sum = np.sum(elig, axis=1)
    most_elig_syn = np.argpartition(elig_sum, -update_partition)[-update_partition:]

    ### Update most_elig_syn blame-wise
    update_new = []
    for i in range(0, len(weight_m)):
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

    weight_m = weight_m + update_new

    return weight_m


def synt_train_Tempotron(path, datamaker):
    ### Training setup
    lr = 0.0002
    to_update = 0.2
    epochs = 5000
    omega_rate = 0.1

    noise = True
    datamaker.bg_freq_rate = 1

    plotting = False

    datamaker.feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()
    # np.save(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea), datamaker.feature_list)

    neuron_A = Tempotron(omega_rate=omega_rate, pre_syn=datamaker.n, memory=100)
    # neuron_A.weight_array = np.load(path + "/weights/Temp_weights_N_{}.npy".format(datamaker.n))

    neuron_A_error = []
    for e in range(0, epochs):
        print("Epoch {}".format(e))

        # if e % 1000 == 0:
        #     print("Epoch {}".format(e))
            # datamaker.bg_freq_rate += 0.1

        data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise, fea_mode=3)
        neuron_A.memory = len(data[1])

        ### Init. history arrays
        neuron_A_state = []
        neuron_A_out = []
        neuron_A.clear()

        ### Present stimulus
        for bin in range(0, len(data[1])):

            out_A = neuron_A.run(data[:, bin])

            neuron_A_state.append(neuron_A.V_t)
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
            # elif fea_order[count] == 2:
            #     desired_state[index[count]:index[count] + T_fea_local] = 3
            # elif fea_order[count] == 3:
            #     desired_state[index[count]:index[count] + T_fea_local] = 4
            # elif fea_order[count] == 4:
            #     desired_state[index[count]:index[count] + T_fea_local] = 5

            index += np.rint(T_fea_local).astype(int)
            count += 1

        desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2
        # desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2 + n_fea_occur[2] * 3 + n_fea_occur[3] * 4
        # + n_fea_occur[4] * 5

        # error_trace = trainer.calc_error(neuron_A.theta, desired_state, neuron_A_state)
        error_trace = trainer.calc_error(neuron_A.theta, desired_state, neuron_A_out)

        # error = np.where(np.array(neuron_A_state) >= neuron_A.theta)[0].__len__() - desired_spikes
        error = np.where(np.array(neuron_A_out) >= neuron_A.theta)[0].__len__() - desired_spikes

        # neuron_A.feedback_weight_update(neuron_A_state, np.rot90(data), error, error_trace, to_update=to_update, lr=lr)
        # neuron_A.feedback_weight_update(neuron_A_out, np.rot90(data), error, error_trace, to_update=to_update, lr=lr)

        neuron_A.feedback_weight_update(neuron_A_state, data, error, error_trace, to_update=to_update, lr=lr)

        # print("Error: {}".format(error))
        neuron_A_error.append(error)

    if plotting:
        ### Plot input data raster
        markers = datamaker.add_marker(time_occur, fea_order, neuron_A.pre_syn - 1, neuron_A.pre_syn / 40)
        myplt.plot_features(markers, data)

        ### Plot reaction
        plt.axhline(y=neuron_A.theta, linestyle="--", color="k")

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
        plt.plot(neuron_A_out, label="A out")
        plt.axhline(y=neuron_A.theta, linestyle="--", color="k")

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
    np.save(path + "/weights/Temp_weights_N_{}.npy".format(neuron_A.pre_syn), neuron_A.weight_array)

