from models.hyst_neuron import HystNeuron
from models.hyst_layer import  HystLayer
import numpy as np
import matplotlib.pyplot as plt
import os


def quality_test(datamaker, pre_syn):
    background = 100
    epochs = 100
    datamaker.bg_freq_rate = 0.5
    cwd = os.path.dirname(__file__)

    responses = [1,2]
    # responses = [1,2,3]

    fea_1 = []
    fea_2 = []
    # fea_3 = []
    fea_null = []

    fea_1_correct = []
    fea_2_correct = []
    # fea_3_correct = []

    # x_axis = [0]
    x = np.linspace(0, 1, 5)
    x_axis = x.tolist()

    a = 0.75

    for s in x_axis:

        neuron_A = HystNeuron(pre_x=pre_syn, pre_y=1, eta=s, a=a)

        # feature_list = np.load("/Users/jf330/newest_results/feature_list_N_{}_fea_{}.npy".format(pre_syn, datamaker.n_fea)).item()
        feature_list = np.load("/Users/jf330/kent_git/HystNeuron/feature_list_N_{}_fea_{}.npy".format(pre_syn, datamaker.n_fea)).item()

        neuron_A.in_weights = np.load("/Users/jf330/kent_git/HystNeuron/results/weights_N_{}_Eta_{}_A_{}_fixed.npy".format(pre_syn, s, a))

        print("Test for Eta: {}, A: {}".format(neuron_A.eta, neuron_A.a))

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

                actual_num_spikes = np.where(np.array(neuron_A_state) >= neuron_A.K)[0].__len__()

                fea_responses.append(actual_num_spikes)
                if actual_num_spikes == desired_spikes:
                    fea_correct += 1

            null_responses = []
            desired_null = 0
            for e in range(0, epochs):
                # print("Epoch {}".format(e))

                data = datamaker.gen_background_data()
                empty_feature_data = np.insert(data, [np.random.randint(0, background)], np.zeros((pre_syn, np.rint(datamaker.T_fea/datamaker.dt).astype(int))), axis=1)

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

                actual_num_spikes = np.where(np.array(neuron_A_state) >= neuron_A.K)[0].__len__()

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

        fea_null.append(sum(null_spike_sum)/(responses.__len__() * epochs))

        fea_1_correct.append(error_dict["feature_0"])
        fea_2_correct.append(error_dict["feature_1"])
        # fea_3_correct.append(error_dict["feature_2"])

    print(fea_null)
    print(fea_1)
    print(fea_2)
    # print(fea_3)

    print(fea_1_correct)
    print(fea_2_correct)
    # print(fea_3_correct)

    # plot_accuracy(fea_null, fea_1, fea_2, fea_3)
    plot_accuracy(fea_null, fea_1=fea_1, fea_2=fea_2)

    # pyplot.scatter(x_axis, fea_3_correct, color="green", marker="x")
    plt.scatter(x_axis, fea_2_correct, color="blue", marker="x")
    plt.scatter(x_axis, fea_1_correct, color="red", marker="x")
    plt.show()

    acc = (np.array(fea_1_correct) + np.array(fea_2_correct)) / 2
    print("Acc: {}, for a: {}".format(acc, neuron_A.a))
    # pyplot.scatter(x_axis, (np.array(fea_1_correct)+np.array(fea_2_correct)+np.array(fea_3_correct))/3, color="black")
    plt.scatter(x_axis, acc, color="black")
    plt.show()

    return acc


def plot_accuracy(fea_null=[], fea_1=[], fea_2=[], fea_3=[]):

    # error_3 = abs(np.array(fea_3) - 3)

    error_2 = abs(np.array(fea_2) - 2)

    error_1 = abs(np.array(fea_1) - 1)

    error_0 = abs(np.array(fea_null) - 0)

    x = np.linspace(0, 1, 5)
    x_axis = x.tolist()

    sum_error = []
    for i in range(0, 5):
        # sum_error.append(error_0[i] + error_1[i] + error_2[i] + error_3[i])
        sum_error.append(error_0[i] + error_1[i] + error_2[i])

    # plt.scatter(x_axis, fea_3, color="green", marker="x")
    plt.scatter(x_axis, fea_2, color="blue", marker="x")
    plt.scatter(x_axis, fea_1, color="red", marker="x")
    plt.scatter(x_axis, fea_null, color="black", marker="x")

    plt.axhline(y=1, linestyle="--", color="red")
    plt.axhline(y=2, linestyle="--", color="blue")
    # pyplot.axhline(y=3, linestyle="--", color="green")

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
    im = ax.imshow(results)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(eta)))
    ax.set_yticks(np.arange(len(a)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(eta)
    ax.set_yticklabels(a)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(a)):
        for j in range(len(eta)):
            text = ax.text(j, i, results[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("Classification accuracy (alpha/eta)")
    fig.tight_layout()

    plt.ylabel('alpha')
    plt.xlabel('eta')

    plt.show()


def quality_test_heatmap(datamaker, iter):
    a = np.linspace(0, 1, iter)
    eta = np.linspace(0, 1, iter)

    heatmap_results = []
    for i in a:  # FIXME do tqdm progress bar
        print("A: {}".format(i))
        acc = quality_test(datamaker, datamaker.n)
        heatmap_results.append(acc)

    plot_heatmap(a, eta, heatmap_results)