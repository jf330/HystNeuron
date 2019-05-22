import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO
import os

from models.Tempotron_neuron import Tempotron

import utils.training as trainer
import utils.plotting as myplt


def gekko_ode_input(path, datamaker):
    cwd = os.path.dirname(__file__)
    noise = True
    datamaker.feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()

    ### Create GEKKO model
    m = GEKKO()
    h = 2000
    K = 1
    eta = 0.9
    a = 0.2
    b = 0.5

    weight_m = np.load(path + "/weights_N_{}_Eta_{}_A_{}_good.npy".format(datamaker.n, eta, a))

    ### Spatio-temporal inputs
    datamaker.seed = 0
    data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise)

    input_data = np.sum(weight_m * data, axis=0)
    print(input_data[35:45])
    # input_data[39] = 0.8
    time = input_data.__len__()

    m.time = np.linspace(0, time-1, time)  # time points
    u = m.Param(value=input_data)

    ### ODE system
    V = m.Var(0.0)
    X = m.Var(0.0)

    m.Equation(
        V.dt() == -((eta * X * V) + ((1-eta) * a * V)) + u
    )

    m.Equation(
        X.dt() == ((V ** h) / (K ** h + V ** h)) - b * X
        # X.dt() == ( / (np.power(K, h) + np.power(V, h))) - b * X
        # X.dt() == (0.5 * (1 + np.tanh(h * (V - K)))) - b * X
    )

    plt.title("Params - h: {}, K: {}, a: {}, b: {}, eta: {}".format(h, K, a, b, eta))

    ### Solve ODE
    m.options.IMODE = 7
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


def gekko_ode_train(path, datamaker):
    cwd = os.path.dirname(__file__)
    noise = True
    datamaker.feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()

    ### Create GEKKO model
    m = GEKKO()
    h = 2000
    K = 1
    eta = 0.9
    a = 0.2
    b = 0.5

    weight_m = np.load(path + "/weights_N_{}_Eta_{}_A_{}_good.npy".format(datamaker.n, eta, a))

    ### Spatio-temporal inputs
    datamaker.seed = 0
    data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise)

    input_data = np.sum(weight_m * data, axis=0)
    print(input_data[35:45])
    # input_data[39] = 0.5
    time = input_data.__len__()

    m.time = np.linspace(0, time-1, time)  # time points
    u = m.Param(value=input_data)

    ### ODE system
    V = m.Var(0.0)
    X = m.Var(0.0)

    m.Equation(
        V.dt() == -((eta * X * V) + ((1-eta) * a * V)) + u
    )

    m.Equation(
        X.dt() == (V**h / (K**h + V**h)) - b * X
        # X.dt() == ( / (np.power(K, h) + np.power(V, h))) - b * X
        # X.dt() == (0.5 * (1 + np.tanh(h * (V - K)))) - b * X
    )

    plt.title("Params - h: {}, K: {}, a: {}, b: {}, eta: {}".format(h, K, a, b, eta))

    ### Solve ODE
    m.options.IMODE = 7
    m.solve()

    ### Plot results
    plt.axhline(y=K, linestyle="--", color="k")
    plt.plot(m.time, X, 'b-', label='H(t)')
    plt.plot(m.time, V, 'r-', label='V(t)')

    plt.ylim((0, 1.3))
    plt.ylabel('V(t)')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()


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

