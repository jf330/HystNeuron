from models.hyst_neuron import HystNeuron
from models.Tempotron_neuron import Tempotron
from models.hyst_layer import  HystLayer

import numpy as np
import matplotlib.pyplot as plt
import os
from models.LiF_neuron import LIFNeuron


def gutig_quality_test(path, datamaker, pre_syn, a=0.5, iter=21):
    background = 100
    epochs = 100
    datamaker.bg_freq_rate = 1
    cwd = os.path.dirname(__file__)

    # responses = [1]
    # responses = [1, 2]
    responses = [1, 2, 3]
    # responses = [1, 2, 3, 4]

    fea_1 = []
    fea_2 = []
    fea_3 = []
    # fea_4 = []
    fea_null = []

    fea_1_correct = []
    fea_2_correct = []
    fea_3_correct = []
    # fea_4_correct = []

    neuron_A = Tempotron(pre_syn=datamaker.n, memory=150)

    neuron_A.weight_array = np.load(path + "/weights/Temp_weights_N_{}.npy".format(pre_syn))
    feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(pre_syn, datamaker.n_fea)).item()

    resp_dict = {}
    error_dict = {}
    null_spike_sum = []
    res_idx = 0
    for fea in feature_list:
        desired_spikes = responses[res_idx]

        fea_responses = []
        null_responses = []
        fea_correct = 0
        insertion_idx = np.random.randint(0, background)
        for e in range(0, epochs):
            print("Epoch {}".format(e))

            data = datamaker.gen_background_data()
            feature_data = np.insert(data, [insertion_idx], feature_list[fea], axis=1)
            # feature_data = np.insert(data, int(background/2), feature_list[fea], axis=1)

            neuron_A_out = []
            neuron_A_state = []
            neuron_A.clear()
            for bin in range(0, len(feature_data[1])):
                # print("Bin {}".format(bin))

                out_A = neuron_A.run(feature_data[:, bin])

                neuron_A_out.append(out_A)
                neuron_A_state.append(neuron_A.V_t)

            # actual_num_spikes = np.where(np.array(neuron_A_state) >= neuron_A.theta)[0].__len__()
            actual_num_spikes = np.where(np.array(neuron_A_out) >= neuron_A.theta)[0].__len__()

            fea_responses.append(actual_num_spikes)
            if actual_num_spikes == desired_spikes:
                fea_correct += 1

        # desired_null = 0
        # for e in range(0, epochs):
            # print("Epoch {}".format(e))

            # data = datamaker.gen_background_data()
            empty_feature_data = np.insert(data, [insertion_idx], np.zeros((pre_syn, np.rint(datamaker.T_fea/datamaker.dt).astype(int))), axis=1)
            # empty_feature_data = np.insert(data, int(background/2), np.zeros((pre_syn, np.rint(datamaker.T_fea/datamaker.dt).astype(int))), axis=1)

            neuron_A_out = []
            neuron_A_state = []
            neuron_A.clear()
            for bin in range(0, len(empty_feature_data[1])):
                # print("Bin {}".format(bin))

                out_A = neuron_A.run(empty_feature_data[:, bin])

                neuron_A_out.append(out_A)
                neuron_A_state.append(neuron_A.V_t)

            # actual_num_spikes = np.where(np.array(neuron_A_state) >= neuron_A.theta)[0].__len__()
            actual_num_spikes = np.where(np.array(neuron_A_out) >= neuron_A.theta)[0].__len__()

            null_responses.append(actual_num_spikes)

        # print("Null, {}: {} spikes".format(fea, sum(null_responses)))
        # print("Feature, {}: {} spikes".format(fea, sum(fea_responses)))

        # null_spike_sum.append(sum(null_responses))

        resp_mean = (sum(fea_responses) - sum(null_responses))/epochs
        # resp_mean = sum(fea_responses)/epochs

        resp_dict.update({fea: resp_mean})
        error_dict.update({fea: fea_correct/epochs})

        res_idx = res_idx + 1

    print("R_mean: {}".format(resp_dict))
    print("Null_mean: {}".format(sum(null_spike_sum)/(responses.__len__() * epochs)))

    fea_1.append(resp_dict["feature_0"])
    fea_2.append(resp_dict["feature_1"])
    fea_3.append(resp_dict["feature_2"])
    # fea_4.append(resp_dict["feature_3"])

    fea_null.append(sum(null_spike_sum)/(responses.__len__() * epochs))

    fea_1_correct.append(error_dict["feature_0"])
    fea_2_correct.append(error_dict["feature_1"])
    fea_3_correct.append(error_dict["feature_2"])
    # fea_4_correct.append(error_dict["feature_3"])

    print(fea_null)
    print(fea_1)
    print(fea_2)
    print(fea_3)
    # print(fea_4)

    print(fea_1_correct)
    print(fea_2_correct)
    print(fea_3_correct)
    # print(fea_4_correct)

    # acc = (np.array(fea_1_correct))
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct)) / 2
    acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct)) / 3
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct) + np.array(fea_4_correct)) / 4
    print("Acc: {}, for N: {}".format(acc, neuron_A.pre_syn))

    return acc


def quality_test(path, datamaker, pre_syn, a=0.8, iter=11):
    background = 100
    epochs = 100
    datamaker.bg_freq_rate = 1
    cwd = os.path.dirname(__file__)
    plotting = False
    dt_scale = 1

    # responses = [1]
    responses = [1, 2]
    # responses = [3, 4]
    # responses = [1, 2, 3]
    # responses = [2, 3, 4]
    # responses = [1, 2, 3, 4]
    # responses = [2,3,4,5,6]
    # responses = [1, 2, 3, 4, 5]
    # responses = [1, 2, 3, 4, 5, 6]

    fea_1 = []
    fea_2 = []
    # fea_3 = []
    # fea_4 = []
    # fea_5 = []
    # fea_6 = []
    fea_null = []

    fea_1_correct = []
    fea_2_correct = []
    # fea_3_correct = []
    # fea_4_correct = []
    # fea_5_correct = []
    # fea_6_correct = []

    # x_axis = [0]
    x = np.linspace(0, 10, iter)
    # x = np.linspace(0, 1, iter)
    x_axis = x.tolist()
    # a = a * dt_scale

    for s in x_axis:
    # for a in x_axis:
    #     neuron_A = HystNeuron(pre_x=pre_syn, pre_y=1, eta=0, a=a)
    #     neuron_A = HystNeuron(pre_x=pre_syn, pre_y=1, eta=s, a=a)
        neuron_A = LIFNeuron(pre_x=pre_syn, pre_y=1, ref_time=s, decay=a)
    #     neuron_A = HystNeuron(pre_x=pre_syn, pre_y=1, eta=0.6, a=0.8, b=s)

        # neuron_A.b = neuron_A.b * dt_scale
        # neuron_A.g = neuron_A.g * dt_scale
        # neuron_A.d = neuron_A.d * dt_scale

        # neuron_A.weight_m = np.load("/Users/jf330/newest_results2/weights_N_{}_Eta_{}_A_{}_newest.npy".format(pre_syn, s, a))
        # feature_list = np.load("/Users/jf330/newest_results2/feature_list_N_{}_fea_{}_01dt.npy".format(pre_syn, datamaker.n_fea)).item()
        # feature_list = np.load("/Users/jf330/newest_results2/feature_list_N_{}_fea_{}_new.npy".format(pre_syn, datamaker.n_fea)).item()

        # neuron_A.weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_output.npy".format(pre_syn, 0, np.around(a, decimals=3)))
        # neuron_A.weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_output.npy".format(pre_syn, s, np.around(a, decimals=3)))
        # feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(pre_syn, datamaker.n_fea)).item()

        neuron_A.weight_m = np.load(path + "/weights/weights_N_{}_Decay_{}_RefPer_{}_Epoch_{}.npy".format(pre_syn, a, s, 29999))
        feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(pre_syn, datamaker.n_fea)).item()

        # print("Test for Eta: {}, A: {}".format(neuron_A.eta, neuron_A.a))
        print("Test for Ref_period: {}, Decay: {}".format(neuron_A.ref_time, neuron_A.decay))

        resp_dict = {}
        error_dict = {}
        null_spike_sum = []
        res_idx = 0
        for fea in feature_list:
            desired_spikes = responses[res_idx]

            fea_responses = []
            null_responses = []

            fea_correct = 0
            insertion_idx = np.random.randint(0, background)
            for e in range(0, epochs):
                # print("Epoch {}".format(e))

                data = datamaker.gen_background_data()
                feature_data = np.insert(data, [insertion_idx], feature_list[fea], axis=1)
                # feature_data = np.insert(data, int(background/2), feature_list[fea], axis=1)

                neuron_A_out = []
                neuron_A_state = []
                neuron_A.clear()
                for bin in range(0, len(feature_data[1])):
                    # print("Bin {}".format(bin))

                    simul_events = np.where(feature_data[:, bin] >= 1)[0]
                    # print("Clock: {}, inputs: {}".format(bin, simul_events))

                    if simul_events.size != 0:
                        neuron_A.event_input(x=simul_events, y=np.zeros(simul_events.__len__()).tolist(), values=np.ones_like(simul_events))
                    out_A = neuron_A.decay_step()

                    neuron_A_out.append(out_A)
                    neuron_A_state.append(neuron_A.state)


                ### Multi-threshold
                # actual_num_spikes = np.where(np.array(neuron_A_out) >= desired_spikes)[0].__len__()
                # actual_num_spikes = np.where(np.array(neuron_A_state) >= desired_spikes)[0].__len__()

                ### Single-threshold
                actual_num_spikes = np.where(np.array(neuron_A_out) >= neuron_A.K)[0].__len__()
                # actual_num_spikes = np.where(np.array(neuron_A_state) >= neuron_A.K)[0].__len__()

                fea_responses.append(actual_num_spikes)

                ### Multi-threshold
                # if actual_num_spikes == 1:
                ### Single-threshold
                if actual_num_spikes == desired_spikes:
                    fea_correct += 1

            # desired_null = 0
            # for e in range(0, epochs):
                # print("Epoch {}".format(e))

                # data = datamaker.gen_background_data()
                empty_feature_data = np.insert(data, [insertion_idx], np.zeros((pre_syn, np.rint(datamaker.T_fea/datamaker.dt).astype(int))), axis=1)
                # empty_feature_data = np.insert(data, int(background/2), np.zeros((pre_syn, np.rint(datamaker.T_fea/datamaker.dt).astype(int))), axis=1)

                neuron_A_out = []
                neuron_A_state = []
                neuron_A.clear()
                for bin in range(0, len(empty_feature_data[1])):
                    # print("Bin {}".format(bin))

                    simul_events = np.where(empty_feature_data[:, bin] >= 1)[0]
                    # print("Clock: {}, inputs: {}".format(bin, simul_events))

                    if simul_events.size != 0:
                        neuron_A.event_input(x=simul_events, y=np.zeros(simul_events.__len__()).tolist(), values=np.ones_like(simul_events))
                    out_A = neuron_A.decay_step()

                    neuron_A_out.append(out_A)
                    neuron_A_state.append(neuron_A.state)

                # actual_num_spikes = np.where(np.array(neuron_A_state) >= neuron_A.K)[0].__len__()
                actual_num_spikes = np.where(np.array(neuron_A_out) >= neuron_A.K)[0].__len__()

                null_responses.append(actual_num_spikes)

            # print("Null, {}: {} spikes".format(fea, sum(null_responses)))
            # print("Feature, {}: {} spikes".format(fea, sum(fea_responses)))

            # null_spike_sum.append(sum(null_responses))

            resp_mean = (sum(fea_responses) - sum(null_responses))/epochs
            # resp_mean = sum(fea_responses)/epochs

            resp_dict.update({fea: resp_mean})
            error_dict.update({fea: fea_correct/epochs})

            res_idx = res_idx + 1

        print("R_mean: {}".format(resp_dict))
        print("Null_mean: {}".format(sum(null_spike_sum)/(responses.__len__() * epochs)))

        fea_1.append(resp_dict["feature_0"])
        fea_2.append(resp_dict["feature_1"])
        # fea_3.append(resp_dict["feature_2"])
        # fea_4.append(resp_dict["feature_3"])
        # fea_5.append(resp_dict["feature_4"])
        # fea_6.append(resp_dict["feature_5"])

        fea_null.append(sum(null_spike_sum)/(responses.__len__() * epochs))

        fea_1_correct.append(error_dict["feature_0"])
        fea_2_correct.append(error_dict["feature_1"])
        # fea_3_correct.append(error_dict["feature_2"])
        # fea_4_correct.append(error_dict["feature_3"])
        # fea_5_correct.append(error_dict["feature_4"])
        # fea_6_correct.append(error_dict["feature_5"])

    print("Resp null {}".format(fea_null))
    print("Resp Fea 1 {}".format(fea_1))
    print("Resp Fea 2 {}".format(fea_2))
    # print("Resp Fea 3 {}".format(fea_3))
    # print("Resp Fea 4 {}".format(fea_4))
    # print("Resp Fea 5 {}".format(fea_5))
    # print("Resp Fea 6 {}".format(fea_6))

    print("Fea 1 acc. {}".format(fea_1_correct))
    print("Fea 2 acc. {}".format(fea_2_correct))
    # print("Fea 3 acc. {}".format(fea_3_correct))
    # print("Fea 4 acc. {}".format(fea_4_correct))
    # print("Fea 5 acc. {}".format(fea_5_correct))
    # print("Fea 6 acc. {}".format(fea_6_correct))

    # acc = np.array(fea_1_correct)
    acc = (np.array(fea_1_correct) + np.array(fea_2_correct)) / 2
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct)) / 3
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct) + np.array(fea_4_correct)) / 4
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct) + np.array(fea_4_correct) + np.array(fea_5_correct)) / 5
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct) + np.array(fea_4_correct) + np.array(fea_5_correct) + np.array(fea_6_correct)) / 6

    # print("Acc: {}, for a: {}".format(acc, neuron_A.a))
    print("Acc: {}, for Decay: {}".format(acc, neuron_A.decay))

    if plotting:
        # plot_accuracy(fea_null, fea_1, fea_2, fea_3, fea_4)
        # plot_accuracy(fea_null, fea_1, fea_2, fea_3)
        # plot_accuracy(fea_null, fea_1=fea_1, fea_2=fea_2)

        # plt.scatter(x_axis, fea_4_correct, color="yellow", marker="x")
        # plt.scatter(x_axis, fea_3_correct, color="green", marker="x")
        # plt.scatter(x_axis, fea_2_correct, color="blue", marker="x")
        plt.scatter(x_axis, fea_1_correct, color="red", marker="x")
        plt.show()

        # pyplot.scatter(x_axis, (np.array(fea_1_correct)+np.array(fea_2_correct)+np.array(fea_3_correct))/3, color="black")
        plt.scatter(x_axis, acc, color="black")
        plt.show()

    return acc


def learning_curves(path, datamaker, pre_syn):
    background = 100
    epochs = 100
    datamaker.bg_freq_rate = 1
    cwd = os.path.dirname(__file__)
    plotting = True

    responses = [1,2]
    # responses = [3, 4]
    # responses = [2, 3, 4, 5]
    # responses = [1,1,1,1,1]

    fea_1 = []
    fea_2 = []
    # fea_3 = []
    # fea_4 = []
    # fea_5 = []
    # fea_6 = []
    fea_null = []

    fea_1_correct = []
    fea_2_correct = []
    # fea_3_correct = []
    # fea_4_correct = []
    # fea_5_correct = []
    # fea_6_correct = []

    # x_axis = [0]
    x = np.linspace(0, 25000, 251)
    # x = np.linspace(0, 1, iter)
    x_axis = x.tolist()

    for e in x_axis:
        neuron_A = HystNeuron(pre_x=pre_syn, pre_y=1, eta=0, a=0.2)

        # neuron_A.weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_output.npy".format(pre_syn, 0, np.around(a, decimals=3)))
        neuron_A.weight_m = np.load(path + "/weights/weights_N_{}_Eta_{}_A_{}_Read_output_Epoch_{}.npy".format(pre_syn, neuron_A.eta, neuron_A.a, int(e)))
        feature_list = np.load(path + "/features/feature_list_N_{}_fea_{}.npy".format(pre_syn, datamaker.n_fea)).item()

        print("Test for Eta: {}, A: {}, Epoch: {}".format(neuron_A.eta, neuron_A.a, e))

        resp_dict = {}
        error_dict = {}
        null_spike_sum = []
        res_idx = 0
        for fea in feature_list:
            desired_spikes = responses[res_idx]

            fea_responses = []
            fea_correct = 0
            for e in range(0, epochs):
                # print("Epoch {}".format(e))

                data = datamaker.gen_background_data()
                feature_data = np.insert(data, [np.random.randint(0, background)], feature_list[fea], axis=1)
                # feature_data = np.insert(data, int(background/2), feature_list[fea], axis=1)

                neuron_A_out = []
                neuron_A_state = []
                neuron_A.clear()
                for bin in range(0, len(feature_data[1])):
                    # print("Bin {}".format(bin))

                    simul_events = np.where(feature_data[:, bin] >= 1)[0]
                    # print("Clock: {}, inputs: {}".format(bin, simul_events))

                    if simul_events.size != 0:
                        neuron_A.event_input(x=simul_events, y=np.zeros(simul_events.__len__()).tolist(), values=np.ones_like(simul_events))
                    out_A = neuron_A.decay_step()

                    neuron_A_out.append(out_A)
                    neuron_A_state.append(neuron_A.state)

                # actual_num_spikes = np.where(np.array(neuron_A_state) >= neuron_A.K)[0].__len__()
                # actual_num_spikes = np.where(np.array(neuron_A_state) >= desired_spikes)[0].__len__()

                # actual_num_spikes = np.where(np.array(neuron_A_out) >= neuron_A.K)[0].__len__()
                actual_num_spikes = np.where(np.array(neuron_A_out) >= desired_spikes)[0].__len__()

                fea_responses.append(actual_num_spikes)
                if actual_num_spikes == desired_spikes:
                    fea_correct += 1

            null_responses = []
            desired_null = 0
            for e in range(0, epochs):
                # print("Epoch {}".format(e))

                data = datamaker.gen_background_data()
                empty_feature_data = np.insert(data, [np.random.randint(0, background)], np.zeros((pre_syn, np.rint(datamaker.T_fea/datamaker.dt).astype(int))), axis=1)
                # empty_feature_data = np.insert(data, int(background/2), np.zeros((pre_syn, np.rint(datamaker.T_fea/datamaker.dt).astype(int))), axis=1)

                neuron_A_out = []
                neuron_A_state = []
                neuron_A.clear()
                for bin in range(0, len(empty_feature_data[1])):
                    # print("Bin {}".format(bin))

                    simul_events = np.where(empty_feature_data[:, bin] >= 1)[0]
                    # print("Clock: {}, inputs: {}".format(bin, simul_events))

                    if simul_events.size != 0:
                        neuron_A.event_input(x=simul_events, y=np.zeros(simul_events.__len__()).tolist(), values=np.ones_like(simul_events))
                    out_A = neuron_A.decay_step()

                    neuron_A_out.append(out_A)
                    neuron_A_state.append(neuron_A.state)

                # actual_num_spikes = np.where(np.array(neuron_A_state) >= 1)[0].__len__()
                actual_num_spikes = np.where(np.array(neuron_A_out) >= neuron_A.K)[0].__len__()

                null_responses.append(actual_num_spikes)

            # print("Null, {}: {} spikes".format(fea, sum(null_responses)))
            # print("Feature, {}: {} spikes".format(fea, sum(fea_responses)))

            # null_spike_sum.append(sum(null_responses))

            # resp_mean = (sum(fea_responses) - sum(null_responses))/epochs
            resp_mean = sum(fea_responses)/epochs

            resp_dict.update({fea: resp_mean})
            error_dict.update({fea: fea_correct/epochs})

            res_idx = res_idx + 1

        print("R_mean: {}".format(resp_dict))
        print("Null_mean: {}".format(sum(null_spike_sum)/(responses.__len__() * epochs)))

        fea_1.append(resp_dict["feature_0"])
        fea_2.append(resp_dict["feature_1"])
        # fea_3.append(resp_dict["feature_2"])
        # fea_4.append(resp_dict["feature_3"])
        # fea_5.append(resp_dict["feature_4"])
        # fea_6.append(resp_dict["feature_5"])

        fea_null.append(sum(null_spike_sum)/(responses.__len__() * epochs))

        fea_1_correct.append(error_dict["feature_0"])
        fea_2_correct.append(error_dict["feature_1"])
        # fea_3_correct.append(error_dict["feature_2"])
        # fea_4_correct.append(error_dict["feature_3"])
        # fea_5_correct.append(error_dict["feature_4"])
        # fea_6_correct.append(error_dict["feature_5"])

    print("Resp null {}".format(fea_null))
    print("Resp Fea 1 {}".format(fea_1))
    print("Resp Fea 2 {}".format(fea_2))
    # print("Resp Fea 3 {}".format(fea_3))
    # print("Resp Fea 4 {}".format(fea_4))
    # print("Resp Fea 5 {}".format(fea_5))
    # print("Resp Fea 6 {}".format(fea_6))

    print("Fea 1 acc. {}".format(fea_1_correct))
    print("Fea 2 acc. {}".format(fea_2_correct))
    # print("Fea 3 acc. {}".format(fea_3_correct))
    # print("Fea 4 acc. {}".format(fea_4_correct))
    # print("Fea 5 acc. {}".format(fea_5_correct))
    # print("Fea 6 acc. {}".format(fea_6_correct))

    # acc = np.array(fea_1_correct)
    acc = (np.array(fea_1_correct) + np.array(fea_2_correct)) / 2
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct)) / 3
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct) + np.array(fea_4_correct)) / 4
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct) + np.array(fea_4_correct) + np.array(fea_5_correct)) / 5
    # acc = (np.array(fea_1_correct) + np.array(fea_2_correct) + np.array(fea_3_correct) + np.array(fea_4_correct) + np.array(fea_5_correct) + np.array(fea_6_correct)) / 6
    print("Acc: {}, for a: {}".format(acc, neuron_A.a))

    if plotting:
        # plot_accuracy(fea_null, fea_1, fea_2, fea_3, fea_4)
        # plot_accuracy(fea_null, fea_1, fea_2, fea_3)
        plot_accuracy(fea_null, fea_1=fea_1, fea_2=fea_2)

        # plt.scatter(x_axis, fea_4_correct, color="yellow", marker="x")
        # plt.scatter(x_axis, fea_3_correct, color="green", marker="x")
        plt.scatter(x_axis, fea_2_correct, color="blue", marker="x")
        plt.scatter(x_axis, fea_1_correct, color="red", marker="x")
        plt.show()

        # pyplot.scatter(x_axis, (np.array(fea_1_correct)+np.array(fea_2_correct)+np.array(fea_3_correct))/3, color="black")
        plt.scatter(x_axis, acc, color="black")
        plt.show()

    return acc


def plot_accuracy(fea_null=[], fea_1=[], fea_2=[], fea_3=[], fea_4=[]):

    # error_4 = abs(np.array(fea_4) - 4)

    # error_3 = abs(np.array(fea_3) - 3)

    error_2 = abs(np.array(fea_2) - 2)

    error_1 = abs(np.array(fea_1) - 1)

    error_0 = abs(np.array(fea_null) - 0)

    x = np.linspace(0, 25000, fea_1.__len__())
    x_axis = x.tolist()

    sum_error = []
    for i in range(0, fea_1.__len__()):
        # sum_error.append(error_0[i] + error_1[i] + error_2[i] + error_3[i] + error_4[i])
        # sum_error.append(error_0[i] + error_1[i] + error_2[i] + error_3[i])
        sum_error.append(error_0[i] + error_1[i] + error_2[i])
        # sum_error.append(error_0[i] + error_1[i])

    # plt.scatter(x_axis, fea_4, color="yellow", marker="x")
    # plt.scatter(x_axis, fea_3, color="green", marker="x")
    plt.scatter(x_axis, fea_2, color="blue", marker="x")
    plt.scatter(x_axis, fea_1, color="red", marker="x")
    plt.scatter(x_axis, fea_null, color="black", marker="x")

    plt.axhline(y=1, linestyle="--", color="red")
    plt.axhline(y=2, linestyle="--", color="blue")
    # plt.axhline(y=3, linestyle="--", color="green")
    # plt.axhline(y=4, linestyle="--", color="yellow")

    plt.axhline(y=0, linestyle="--", color="k")

    plt.ylabel('Neural response')
    plt.xlabel('S')
    plt.show()

    plt.scatter(x_axis, sum_error, color="black", marker="x")
    plt.ylabel('Error')
    plt.xlabel('S')
    plt.show()

    print(sum_error)


def plot_heatmap(a, eta, results):

    fig, ax = plt.subplots()
    im = ax.imshow(results, vmin=0, vmax=1)

    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel(ylabel="Avg. class. acc.", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(eta)))
    ax.set_yticks(np.arange(len(a)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(np.around(eta, decimals=2))
    ax.set_yticklabels(np.around(a, decimals=2))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ##Loop over data dimensions and create text annotations.
    for i in range(len(a)):
        for j in range(len(eta)):
            text = ax.text(j, i, np.around(results[i][j], decimals=2),
                           ha="center", va="center", color="w")
    ax.set_title("Classification accuracy (alpha/eta)")
    # ax.set_title("Classification accuracy (beta/eta)")
    fig.tight_layout()

    # plt.ylabel('alpha')
    # plt.xlabel('eta')

    plt.xlabel('ref_period')
    plt.ylabel('decay')

    plt.show()


def test_quality_heatmap(path, datamaker, iter):

    a = np.linspace(0, 1, iter)
    eta = np.linspace(0, 1, iter)

    heatmap_results = []
    for i in a:
        print("A: {}".format(i))
        acc = quality_test(path, datamaker, datamaker.n, np.around(i, decimals=3), iter)
        heatmap_results.append(acc)

    np.save(path + "/AccData_iter{}_N{}_Nfea{}_new.npy".format(iter, datamaker.n, datamaker.n_fea), heatmap_results)
    # heatmap_results = np.load(path + "/AccData_iter{}_N{}_Nfea{}.npy".format(iter, datamaker.n, datamaker.n_fea))

    plot_heatmap(a, eta, heatmap_results)


def load_quality_heatmap(path, datamaker, iter):

    a = np.linspace(0, 1, iter)
    eta = np.linspace(0, 10, iter)

    heatmap_results = np.load(path + "/AccData_iter{}_N{}_Nfea{}_new.npy".format(iter, datamaker.n, datamaker.n_fea))

    plot_heatmap(a, eta, heatmap_results)
