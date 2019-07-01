import tests
import utils.accuracy_tests
import utils.other_tests
from utils.datamaker import Datamaker
import os

import matplotlib.pyplot as plt
# plt.switch_backend("agg")

import time


def main(
        model_type,
        test_type,
        iterations,
        path,
):
    # with np.errstate(divide='ignore'): # Ignore overflow, divide by 0 etc. warning messages
    # if pwd.getpwuid(os.getuid())[0] == "jf330":

    if path == "default":
        # cwd = os.path.dirname(__file__)  # Works for PyCharm
        cwd = os.getcwd()  # Works for Myrtle
        path = cwd + "/results/106"

    dt_scale = 1
    n = 100  # Number of neurons
    dt = 0.001 * dt_scale  # Bin length (s)
    duration = 0.2  # Trial duration background (s)
    n_fea = 3  # Total number of features and distractors
    cf_mean = 2  # Mean number of occurrences for each feature
    T_fea = 0.05  # Base feature duration (s)
    fr = 5  # Background spiking frequency (Hz)
    random_seed = 0  # Start random seed

    datamaker = Datamaker(n, duration, dt, n_fea, cf_mean, T_fea, fr, random_seed, ["random", "random"])
    start_time = time.time()

    if test_type == "cont_curr_input":
        tests.cont_current_input()
    elif test_type == "aedat_input":
        tests.aedat_input()
    elif test_type == "simple_input":
        tests.simple_input()
    elif test_type == "synt_input":
        tests.synt_input(path, datamaker)
    elif test_type == "synt_train_many":
        tests.synt_train_many(path, datamaker, iterations, dt_scale)
    elif test_type == "synt_train":
        tests.synt_train(path, datamaker, dt_scale)
    elif test_type == "synt_train_Tempotron":
        utils.other_tests.synt_train_Tempotron(path, datamaker)
    elif test_type == "synt_train_bp":
        tests.synt_train_bp(path, datamaker)
    elif test_type == "aedat_train":
        tests.aedat_train(path, datamaker)
    elif test_type == "quality_test":
        utils.accuracy_tests.quality_test(path, datamaker, n)
    elif test_type == "learning_curves":
        utils.accuracy_tests.learning_curves(path, datamaker, n)
    elif test_type == "gutig_quality_test":
        utils.accuracy_tests.gutig_quality_test(path, datamaker, n)
    elif test_type == "test_quality_heatmap":
        utils.accuracy_tests.test_quality_heatmap(path, datamaker, iterations)
    elif test_type == "load_quality_heatmap":
        utils.accuracy_tests.load_quality_heatmap(path, datamaker, iterations)
    elif test_type == "gekko_test":
        utils.other_tests.gekko_ode_input(path, datamaker)
    elif test_type == "gekko_ode_train":
        utils.other_tests.gekko_ode_train(path, datamaker)
    elif test_type == "run_matlab":
        utils.other_tests.run_matlab(path)
    elif test_type == "train_matlab":
        utils.other_tests.train_matlab(path, datamaker)
    elif test_type == "train_lif":
        utils.other_tests.train_lif(path, datamaker)
    elif test_type == "lif_train_many":
        utils.other_tests.lif_train_many(path, datamaker, iterations)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    import argparse

    text_type = str

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-m', '--model_type',
        default='HystNeuron',
        type=text_type,
        help='Spiking neuron type',
    )
    parser.add_argument(
        '--test_type',
        default='',
        type=text_type,
        help='Test type',
    )
    parser.add_argument(
        '--iter',
        default=1,
        type=int,
        help='Number of parameter iterations',
    )
    parser.add_argument(
        '--path',
        default='default',
        type=text_type,
        help='Default directory path',
    )

    args, unknown = parser.parse_known_args()
    try:
        main(
            args.model_type,
            args.test_type,
            args.iter,
            args.path,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
