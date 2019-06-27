import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO
import os
import scipy.io as sio
import matlab.engine

from models.Tempotron_neuron import Tempotron

import utils.training as trainer
import utils.plotting as myplt


def train_matlab(path, datamaker):
    ### Training setup
    # lr = 0.0005
    lr = 0.002
    to_update = 0.2
    epochs = 1500
    omega_rate = 0.3
    momentum = 0.2

    noise = True
    datamaker.bg_freq_rate = 1
    plotting = True
    readout = "output"
    matlab_path = path + "/matlab_data/"

    eng = matlab.engine.start_matlab()

    h = 250
    K = 1
    eta = 0.9
    a = 0.3
    b = 0.3
    d = 1
    g = 1

    params = {"a": a, "b": b, "g": g, "d": d, "eta": eta, "K": K, "h": h}
    sio.savemat(matlab_path + "N_" + str(datamaker.n) + "_params.mat", params)

    ### Load pre-generated features
    features_path = path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)
    if os.path.isfile(features_path):
        datamaker.feature_list = np.load(features_path, allow_pickle=True).item()
    else:
        np.save(features_path, datamaker.feature_list, allow_pickle=True)

    ### Load pre-trained weights or random init.
    # weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_cont.npy".format(datamaker.n, eta, np.around(a, decimals=3)))
    weight_m = np.random.rand(datamaker.n) * omega_rate
    update_prev = np.zeros((datamaker.n))

    neuron_A_error = []
    fea1_spike_est_arr = []
    fea2_spike_est_arr = []
    fea3_spike_est_arr = []
    for e in range(0, epochs):
        print("Epoch {}".format(e))

        if e % 50 == 0:
            np.save(path + "/weights/weights_N_{}_Eta_{}_A_{}_Epoch_{}_cont.npy".format(datamaker.n, eta, np.around(a, decimals=3), e), weight_m)

        data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise, fea_mode=3)
        for neur_idx in range(0, datamaker.n):
            input_data = weight_m[neur_idx] * data[neur_idx]

            # matlab_path = matlab_data + "test"

            sio.savemat(matlab_path + "N_" + str(datamaker.n) + "_weight_inputs_" + str(neur_idx) + ".mat",
                        {"data": abs(input_data[np.where(input_data != 0)])})
                        # {"data": input_data[np.where(input_data != 0)]})
            sio.savemat(matlab_path + "N_" + str(datamaker.n) + "_time_inputs_" + str(neur_idx) + ".mat",
                        {"data": np.where(input_data != 0)})

        stop_time = len(data[0, :])
        outputs = eng.sim_matlab(stop_time, datamaker.n, matlab_path, nargout=2)

        V = outputs[0][0]
        R = outputs[0][1]
        O = outputs[0][2]
        S = outputs[0][3]
        time = outputs[0][4]
        elig = outputs[1][0]

        spike_est = np.trapz(S, time)

        ### Weight update
        desired_state = np.zeros(len(data[1]))
        count = 0
        fea1_spike_est = []
        fea2_spike_est = []
        fea3_spike_est = []
        index = np.rint(time_occur / datamaker.dt).astype(int)
        while count < len(index):
            fea = datamaker.feature_list['feature_' + str(fea_order[count])]
            T_fea_local = fea[0].size

            s_idx = find_nearest(time, index[count])
            e_idx = find_nearest(time, index[count] + T_fea_local)
            fea_spike_integral = np.trapz(S[s_idx:e_idx], time[s_idx:e_idx])
            # print(f"feature {i}, spike_est {fea_spike_est}")

            if fea_order[count] == 0:
                desired_state[index[count]:index[count] + T_fea_local] = 1
                fea1_spike_est.append(fea_spike_integral)
            elif fea_order[count] == 1:
                desired_state[index[count]:index[count] + T_fea_local] = 2
                fea2_spike_est.append(fea_spike_integral)
            elif fea_order[count] == 2:
                desired_state[index[count]:index[count] + T_fea_local] = 3
                fea3_spike_est.append(fea_spike_integral)
            # elif fea_order[count] == 3:
            #     desired_state[index[count]:index[count] + T_fea_local] = 4
            # elif fea_order[count] == 4:
            #     desired_state[index[count]:index[count] + T_fea_local] = 5
            # elif fea_order[count] == 5:
            #     desired_state[index[count]:index[count] + T_fea_local] = 6

            index += np.rint(T_fea_local).astype(int)
            count += 1

        # desired_spikes = n_fea_occur[0] * 1
        # desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2
        desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2 + n_fea_occur[2] * 3

        error = spike_est - desired_spikes

        update_new = feedback_weight_update(weight_m, elig, error, to_update=to_update, lr=lr)

        update_all = np.array(update_new) + (update_prev * momentum)
        update_prev = np.array(update_new)
        weight_m = weight_m + update_all
        weight_m = np.clip(weight_m, 0, 1)

        print("Error: {}, Desired: {}".format(error, desired_spikes))
        neuron_A_error.append(error)

        if len(fea1_spike_est) != 0:
            fea1_spike_est_arr.append(sum(fea1_spike_est)/len(fea1_spike_est))
        else:
            fea1_spike_est_arr.append(None)

        if len(fea2_spike_est) != 0:
            fea2_spike_est_arr.append(sum(fea2_spike_est)/len(fea2_spike_est))
        else:
            fea2_spike_est_arr.append(None)

        if len(fea3_spike_est) != 0:
            fea3_spike_est_arr.append(sum(fea3_spike_est)/len(fea3_spike_est))
        else:
            fea3_spike_est_arr.append(None)

    if plotting:
        plt.axhline(y=0, linestyle="--", color="k")
        plt.scatter(np.arange(0, len(neuron_A_error)), neuron_A_error, marker="x", label='Training Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.show()

        ### Plot input data raster
        markers = datamaker.add_marker(time_occur, fea_order, datamaker.n, datamaker.n / 40)
        myplt.plot_features(markers, data)

        ### Plot results
        plt.title("Params - h: {}, K: {}, a: {}, b: {}, d: {}, g: {}, eta: {}".format(h, K, a, b, d, g, eta))
        markers = datamaker.add_marker(time_occur, fea_order, 1.2, 0.05)
        i = 0
        for marker in markers:
            plt.gca().add_patch(marker)
            plt.axvspan(marker.get_x(), marker.get_x() + marker.get_width(), alpha=0.2, color="black")
            plt.gca().add_patch(marker)

            s_idx_final = find_nearest(time, marker.get_x())
            e_idx_final = find_nearest(time, marker.get_x() + marker.get_width())
            fea_spike_est_final = np.trapz(S[s_idx_final:e_idx_final], time[s_idx_final:e_idx_final])
            print(f"Final epoch: feature {i}, spike_est {fea_spike_est_final}")
            # plt.gca().text(marker.get_x(), marker.get_x() + marker.get_width(), str(fea_spike_est))

            i += 1

        plt.axhline(y=K, linestyle="--", color="k")
        # plt.plot(time, R, 'r-', label='R(t)')
        # plt.plot(time, O, 'y-', label='O(t)')
        plt.plot(time, S, 'g-', label='S(t)')
        plt.plot(time, V, 'b-', label='V(t)')

        # plt.ylim((0, 1.3))
        plt.ylabel('V(t)')
        plt.xlabel('time')
        plt.legend(loc='best')
        plt.show()

        ### Plot average feature spike estimate
        plt.scatter(np.arange(0, len(fea3_spike_est_arr)), fea3_spike_est_arr, color="green", marker="x")
        plt.scatter(np.arange(0, len(fea2_spike_est_arr)), fea2_spike_est_arr, color="blue", marker="x")
        plt.scatter(np.arange(0, len(fea1_spike_est_arr)), fea1_spike_est_arr, color="red", marker="x")

        plt.axhline(y=1, linestyle="--", color="red")
        plt.axhline(y=2, linestyle="--", color="blue")
        plt.axhline(y=3, linestyle="--", color="green")
        plt.show()

    ### Save final trained weights
    np.save(path + "/weights/weights_N_{}_Eta_{}_A_{}_Epoch_{}_cont.npy".format(datamaker.n, eta, np.around(a, decimals=3), e), weight_m)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def run_matlab(path):
    eng = matlab.engine.start_matlab()
    outputs = eng.sim_matlab()


def gekko_ode_input(path, datamaker):
    cwd = os.path.dirname(__file__)
    noise = False
    datamaker.feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(datamaker.n, datamaker.n_fea)).item()

    ### Create GEKKO model
    m = GEKKO()
    h = 250
    K = 1
    eta = 0.9
    a = 1.0
    b = 0.5
    d = 1
    g = 1

    weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_output.npy".format(datamaker.n, eta, a))

    # a = a/2
    # b = b/2
    # d = d/2
    # g = g/2

    ### Spatio-temporal inputs
    datamaker.seed = 0
    data, time_occur, fea_order, n_fea_occur, fea_time, fea_order = datamaker.gen_input_data(noise=noise)

    input_data = np.sum(weight_m * data, axis=0)
    sio.savemat("/Users/jf330/Desktop/weight_inputs.mat", {"data": abs(input_data[np.where(input_data!=0)])})
    sio.savemat("/Users/jf330/Desktop/time_inputs.mat", {"data": np.where(input_data!=0)})

    # input_data = np.sum(data, axis=0)
    # print(input_data[35:45])
    # input_data[39] = 0.8
    # time = input_data.__len__()
    time = 100
    # time = 100*2
    # input_data = np.zeros(time+1)
    # input_data[1] = 0
    # input_data[2] = 1
    m.time = np.linspace(0, time, time+1)  # time points
    u = m.Param(value=input_data[0:time+1])

    ### ODE system
    V = m.Var(0.0)
    X = m.Var(0.0)

    m.Equation(
        V.dt() == -((eta * X * V * g) + ((1-eta) * a * V)) + u
    )

    m.Equation(
        X.dt() == ((V ** h) / (K ** h + V ** h)) * d  - b * X
        # X.dt() == (np.power(V, h) / (np.power(K, h) + np.power(V, h))) - b * X
        # X.dt() == (0.5 * (1 + np.tanh(h * (V - K)))) - b * X
    )

    plt.title("Params - h: {}, K: {}, a: {}, b: {}, d: {}, eta: {}".format(h, K, a, b, d, eta))

    ### Solve ODE
    m.options.IMODE = 9
    m.solve()

    print("Max X: {}, Max V: {}".format(max(X), max(V)))

    ### Plot results
    plt.axhline(y=K, linestyle="--", color="k")
    plt.plot(m.time, X, 'b-', label='H(t)')
    plt.plot(m.time, V, 'r-', label='V(t)')

    plt.ylim((-2, 1.3))
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
        g = 1

        m.time = np.linspace(0, input_data.__len__() - 1, input_data.__len__())  # time points
        u = m.Param(value=input_data)

        ### ODE system
        V = m.Var(0.0)
        R = m.Var(0.0)

        m.Equation(V.dt() == -((eta * R * V * g) + ((1 - eta) * a * V)) + u)
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
            update_new = feedback_weight_update(weight_m, neuron_A_out, synt_reset, error, to_update=to_update, lr=lr)

            weight_m = weight_m + update_new

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


def feedback_weight_update(weight_m, elig_sum, error, to_update=0.2, lr=0.001):

    ### Select top x% most eligible pre-syn neurons over whole input
    update_partition = np.rint((weight_m.__len__()) * to_update).astype(int)

    most_elig_syn = np.argpartition(elig_sum, -update_partition)[-update_partition:]

    ### Update most_elig_syn blame-wise
    update_new = []
    for i in range(0, len(weight_m)):
        if i in most_elig_syn:
            update_new.append(error * -lr)
            # update_new.append(-elig_sum[i] * lr)

            # if error > 0:
            #     update_new.append(-lr)
            # elif error < 0:
            #     update_new.append(lr)
            # else:
            #     update_new.append(0)
        else:
            update_new.append(0)

    return update_new


def synt_train_Tempotron(path, datamaker):
    ### Training setup
    lr = 0.0001
    to_update = 0.1
    epochs = 20000
    omega_rate = 0.2

    noise = True
    datamaker.bg_freq_rate = 1

    plotting = True

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
            elif fea_order[count] == 2:
                desired_state[index[count]:index[count] + T_fea_local] = 3
            # elif fea_order[count] == 3:
            #     desired_state[index[count]:index[count] + T_fea_local] = 4
            # elif fea_order[count] == 4:
            #     desired_state[index[count]:index[count] + T_fea_local] = 5

            index += np.rint(T_fea_local).astype(int)
            count += 1

        # desired_spikes = n_fea_occur[0] * 1
        desired_spikes = n_fea_occur[0] * 1 + n_fea_occur[1] * 2 + n_fea_occur[2] * 3
        # + n_fea_occur[3] * 4
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

