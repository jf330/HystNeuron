import numpy as np
import matplotlib.patches as patches


class Datamaker(object):
    def __init__(
            self,
            n,
            duration,
            dt,
            n_fea,
            cf_mean,
            T_fea,
            fr,
            seed,
            feature_mode=[]
    ):
        self.n = n
        self.duration = duration
        self.dt = dt
        self.n_fea = n_fea
        self.cf_mean = cf_mean
        self.T_fea = T_fea
        self.fr = fr
        self.seed = seed
        self.bg_freq_rate = 1

        if feature_mode == []:
            # self.feature_list = self.gen_features()
            self.feature_list = self.gen_features_kron_noisy()
        else:
            self.feature_list = self.gen_features_var(feature_mode[0], feature_mode[1])

    def gen_input_data(self, noise=False, fea_mode=3):
        np.random.seed(self.seed)

        syn_ratio = 0.0001
        if noise:
            #  Random noisy background
            data = self.gen_background_data()
        else:
            #  Empty background
            data = np.zeros((self.n, np.rint(self.duration / self.dt).astype(int)))

        #  No background
        # data = np.zeros((self.n, 1))

        if fea_mode == 1:
            n_fea_occur1 = np.random.poisson(self.cf_mean, self.n_fea)  # TODO Can I reduce that, return 1 thing?
            choice = np.random.choice(self.n_fea-1,1)
            n_fea_occur = np.zeros_like(n_fea_occur1)
            n_fea_occur[choice] = n_fea_occur1[choice]  # Only a single feature
        elif fea_mode == 2:
            n_fea_occur1 = np.random.poisson(self.cf_mean, self.n_fea)
            n_fea_occur = np.ones_like(n_fea_occur1)  # One of each features
        elif fea_mode == 3:
            n_fea_occur = np.random.poisson(self.cf_mean, self.n_fea)

        time_occur = self.time_occurrence(n_fea_occur)
        # time_occur = np.arange(0, sum(n_fea_occur), 1) * self.dt
        fea_order = self.gen_order_occur(n_fea_occur)

        index = np.rint(time_occur / self.dt).astype(int)

        feature_data = data
        fea_time = []
        count = 0

        # FOR FEATURES OF VARIED LENGTH
        while count < len(index):
            fea_time.append(index[count])
            fea = self.feature_list['feature_' + str(fea_order[count])]
            T_fea_local = fea[0].size
            # print('T_fea_local', T_fea_local)
            feature_data = np.insert(feature_data, index[count], fea.T, axis=1)
            index += np.rint(T_fea_local).astype(int)
            count += 1

        self.seed += 1
        return feature_data, time_occur, fea_order, n_fea_occur, fea_time, fea_order

    def gen_unweighted_inputs(self, data, syn_ratio=0.001):
        datalen = data.shape[1] + 100
        # Create the synaptic kernel, and then remove the negligible tail
        syn_kernel = self.kernel_fn(len(data[0]), 0)
        syn_memory_len = self.get_memory_len(syn_kernel, syn_ratio)
        # synaptic memory length
        syn_kernel = syn_kernel[:syn_memory_len]

        # Precompute the unweighted input - this is unaffected by a change in weights
        # Ultimately this should be returned by gen_input_data along with, or instead of, 'data'.
        presyn_input = np.zeros((data.shape[0], datalen))
        for neuron, ith_bin in zip(*np.where(data)):
            mem_len = min(syn_memory_len, datalen - ith_bin)
            presyn_input[neuron, ith_bin:ith_bin + mem_len] += syn_kernel[:mem_len]
        return presyn_input

    def gen_background_data(self, duration=0):
        if duration == 0:
            duration = self.duration

        gen_bg = np.random.random((self.n, np.rint(duration / self.dt).astype(int))) < self.fr * self.bg_freq_rate * self.dt
        gen_bg = gen_bg.astype(int)
        return gen_bg

    def time_occurrence(self, n_fea_occur):
        return np.sort(np.round(np.random.random(np.sum(n_fea_occur)) * self.duration, 3))

    def gen_order_occur(self, n_fea_occur):
        count = 0
        order = []
        while count < self.n_fea:
            order += [count] * n_fea_occur[count]
            count += 1
        return np.random.permutation(order)

    def gen_features(self):
        a = 0
        features = {}
        while a < self.n_fea:
            features['feature_' + str(a)] = (np.random.random(
                (self.n, np.rint(self.T_fea / self.dt).astype(int))) < self.fr * self.dt).astype(int)
            a += 1
        return features

    def gen_features_var(self, option1, option2):
        a = 0
        features = {}
        while a < self.n_fea:
            if (a+1) % 2 != 0:  # First predefined feature

                if option1 == "random":  # RANDOM TIMES, RANDOM NEURONS
                    features['feature_'+str(a)] = (np.random.random(
                        (self.n, np.rint(self.T_fea/self.dt).astype(int))) < self.fr* self.dt).astype(int)

                elif option1 == "randomAll":  # RANDOM TIMES, ALL NEURONS
                    features['feature_' + str(a)] = np.zeros((self.n, np.rint(self.T_fea / self.dt).astype(int))).astype(int)
                    for j in range(self.n):
                        one = np.random.randint(np.rint(self.T_fea / self.dt).astype(int))
                        features['feature_' + str(a)][j][one] = 1

                elif option1 == "fixed":  # FIXED TIMES, FIXED NEURONS
                    features['feature_' + str(a)] = np.zeros((self.n, np.rint(self.T_fea / self.dt).astype(int))).astype(int)
                    features['feature_' + str(a)][0][2] = 1
                    features['feature_' + str(a)][1][10] = 1
                    features['feature_' + str(a)][2][20] = 1

                elif option1 == "freq":
                    frequencies = [15, 12, 9, 6]  # Example feature activation frequency list
                    features['feature_' + str(a)] = (np.random.random(
                        (self.n, np.rint(self.T_fea / self.dt).astype(int))) < frequencies[a] * self.dt).astype(int)

            else:  # Than modified feature
                if option2 == 'fixed':
                    features['feature_' + str(a)] = np.zeros((self.n, np.rint(self.T_fea / self.dt).astype(int))).astype(int)
                    features['feature_' + str(a)][3][15] = 1
                    features['feature_' + str(a)][4][30] = 1

                if option2 == "random":  # RANDOM TIMES, RANDOM NEURONS
                    features['feature_'+str(a)] = (np.random.random(
                        (self.n, np.rint(self.T_fea/self.dt).astype(int))) < self.fr * self.dt).astype(int)

                elif option2 == "stretch":  # Than feature stretched by Kronecker product (np.kron)
                    # features['feature_'+str(a)] = np.kron(features['feature_'+str(a - 1)], [0, 0, 1, 0])
                    # features['feature_'+str(a)] = np.kron(features['feature_'+str(a - 1)], [0, 1, 0])
                    features['feature_'+str(a)] = np.kron(features['feature_'+str(a - 1)], [0, 1])

                elif option2 == "flip":  # Than invert the feature (np.flip)
                    features['feature_' + str(a)] = np.flip(features['feature_' + str(a - 1)], 1)

                elif option2 == "freq":  # Than base feature on next freq. in the list
                    features['feature_' + str(a)] = (np.random.random(
                        (self.n, np.rint(self.T_fea / self.dt).astype(int))) < frequencies[a] * self.dt).astype(int)
            a += 1
        return features

    def gen_features_kron_noisy(self):
        a = 0
        features = {}
        while a < self.n_fea:
            if (a + 1) % 2 != 0:
                features['feature_' + str(a)] = (np.random.random(
                    (self.n, np.rint(self.T_fea / self.dt).astype(int))) < self.fr * self.dt).astype(int)
            else:
                # fea_kron = np.kron(features['feature_'+str(a-1)], [0,0,1,0])
                # fea_kron = np.kron(features['feature_'+str(a-1)], [0,1,0])
                fea_kron = np.kron(features['feature_' + str(a - 1)], [0, 1])
                fea_rand = features['feature_' + str(a)] = (np.random.random(
                    (self.n, np.rint(self.T_fea * 2 / self.dt).astype(int))) < self.fr * self.dt).astype(int)
                for i in range(fea_kron.shape[0]):
                    sum_n = 0
                    for n in np.where(fea_kron[i])[0]:
                        if fea_rand[i, n] != 1:
                            sum_n += n
                        else:
                            fea_rand[i, n] = 0
                    row_where = np.where(fea_rand[i])[0]
                    if row_where.size != 0:
                        for z in range(sum_n):
                            fea_rand[i, np.random.choice(row_where)] = 0
                fea = fea_kron + fea_rand
                features['feature_' + str(a)] = fea
            a += 1

        return features

    def add_fea_old(self, n_inputs, fea_input):
        T_fea = self.T_fea

        inputs = np.copy(n_inputs)
        start = int((n_inputs.shape[2] - T_fea * 1000) / 2)
        for n in inputs:
            n[:, start:start + fea_input.shape[1]] += fea_input
        return inputs

    def add_fea(self, n_inputs, fea_input, T_fea_local):
        inputs = np.copy(n_inputs)
        start = int((n_inputs.shape[2] - T_fea_local) / 2)
        for n in inputs:
            n[:, start:start + fea_input.shape[1]] += fea_input
        return inputs

    def add_marker_old(self, time_occur, feature_order, marker_y, marker_height):

        index = time_occur / self.dt
        markers = []
        color = ['r', 'b', 'g', 'm', '#FF6600', '#00ffff', '#FDEE00', '#D71868', 'y', 'c', 'k']
        count = 0
        while count < len(index):
            index[count] += count * self.T_fea / self.dt
            markers.append(patches.Rectangle((index[count] - 0.5, marker_y), self.T_fea / self.dt, marker_height,
                                             fc=color[feature_order[count]], alpha=0.6, ec='#000000'))
            count += 1
        return markers

    def add_marker(self, time_occur, feature_order, marker_y, marker_height):

        index = np.rint(time_occur / self.dt).astype(int)
        markers = []
        color = ['r', 'b', 'g', 'm', '#FF6600', '#00ffff', '#FDEE00', '#D71868', 'y', 'c', 'k']
        count = 0
        # FOR PROLONGED FEATURES
        while count < len(index):
            fea = self.feature_list['feature_' + str(feature_order[count])]
            T_fea_local = fea[0].size
            markers.append(patches.Rectangle((index[count] - 0.5, marker_y), T_fea_local, marker_height,
                                             fc=color[feature_order[count]], alpha=0.6, ec='#000000'))
            index += np.rint(T_fea_local).astype(int)
            count += 1
        return markers

