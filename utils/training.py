import numpy as np


def calc_hyst(self, input_data):
    # Uniform out function based input
    hyst_data = []
    for bin in range(0, len(input_data[1])):
        input_slice = input_data[:, bin]
        if bin != 0:
            hyst_data.append(input_slice + self.b * hyst_data[bin - 1])
        else:
            hyst_data.append(input_slice)

    return np.array(hyst_data)


def calc_integrals(threshold, target_state, state, fea_order, time_occur, datamaker):

    state = np.array(state)
    spikes = np.where(state >= threshold)[0].__len__()
    error_bg = spikes

    error_trace = np.zeros(state.__len__())
    error = 0
    count = 0
    index = np.rint(time_occur / datamaker.dt).astype(int)
    while count < len(index):
        fea = datamaker.feature_list['feature_' + str(fea_order[count])]
        T_fea_local = fea[0].size
        seg_state = state[index[count]:index[count] + T_fea_local]
        seg_target = target_state[index[count]]

        actual_spikes = np.where(seg_state >= threshold)[0].__len__()
        error_seg = np.sum(seg_state[np.where(seg_state >= threshold)]) - seg_target
        # error_seg = (actual_spikes-seg_target)
        # error_seg = np.sum(seg_state[np.where(seg_state >= seg_target)]) - seg_target
        error_bg = error_bg - actual_spikes
        error_trace[index[count]:index[count] + T_fea_local] = error_seg

        error = error + error_seg

        index += np.rint(T_fea_local).astype(int)
        count += 1

    error_trace[np.where(error_trace == 0)] = error_bg/50  # TODO should be normalized, too large error trace for background
    error = error + error_bg

    return error, error_trace



