import tests
import utils.accuracy_tests
from utils.datamaker import Datamaker

import time

def main(
        model_type,
        test_type,
        iterations,
):
    # with np.errstate(divide='ignore'): # Ignore overflow, divide by 0 etc. warning messages

    n = 300  # Number of neurons
    dt = 0.001  # Bin length (s)
    duration = 0.1  # Trial duration background (s)
    n_fea = 2  # Total number of features and distractors
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
        tests.synt_input(datamaker)
    elif test_type == "synt_train_many":
        tests.synt_train_many(datamaker, iterations)
    elif test_type == "synt_train":
        tests.synt_train(datamaker)
    elif test_type == "synt_train_bp":
        tests.synt_train_bp(datamaker)
    elif test_type == "aedat_train":
        tests.aedat_train(datamaker)
    elif test_type == "quality_test":
        utils.accuracy_tests.quality_test(datamaker, n)

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

    args, unknown = parser.parse_known_args()
    try:
        main(
            args.model_type,
            args.test_type,
            args.iter,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
