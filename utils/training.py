import numpy as np


def calc_synt_reset(input_data, decay):
    # Uniform out function based input
    hyst_data = []
    for bin in range(0, len(input_data[1])):
        input_slice = input_data[:, bin]
        if bin != 0:
            hyst_data.append(input_slice + decay * hyst_data[bin - 1])
        else:
            hyst_data.append(input_slice)

    return np.array(hyst_data)


def calc_integrals_synt(threshold, target_state, state, fea_order, time_occur, datamaker):

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

    error_trace[np.where(error_trace == 0)] = 0.2  # TODO should be normalized, too large error trace for background
    error = error + error_bg

    return error, error_trace


def calc_error(threshold, desired, actual):
    error = np.zeros((len(desired)))
    target = 0
    target_value = 0
    pattern_on = False
    for bin in range(0, len(desired)):
        target = desired[bin]
        if target != 0 and pattern_on == False:
            pattern_on = True
            target_index_start = bin
            target_value = target
        if (target == 0) and (target_value != 0):
            target_index_end = bin - 1

            range_len = target_index_end - target_index_start
            segment = np.array(actual[target_index_start:target_index_end])

            ### Single-threshold error
            actual_spikes = np.where(segment >= threshold)[0].__len__()
            error_value = actual_spikes - target_value

            ### Multi-threshold error
            # actual_spikes = np.where(segment >= target_value)[0].__len__()
            # error_value = actual_spikes - 1

            ### Supra-threshold integral error
            # actual_output = np.sum(segment[np.where(segment >= threshold)])
            # error_value = actual_output - target_value

            if error_value == 0:
                error[target_index_start:target_index_end] = np.zeros((range_len)).tolist()
            else:
                error_range = np.ones((range_len)) * error_value
                error[target_index_start:target_index_end] = error_range.tolist()

            target_value = 0
            pattern_on = False
            # error[bin] = 0
        if (target == 0) and (target_value == 0):
            if actual[bin] >= threshold:
                error[bin] = threshold
            else:
                error[bin] = 0

    return error.tolist()
