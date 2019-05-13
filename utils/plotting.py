import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


def plot_features(markers, input_data):
    plt.rcParams['figure.figsize'] = (15.0, 7.5)
    y, x = np.argwhere(input_data == 1).T
    plt.scatter(x, y, s=1)

    for marker in markers:
        plt.gca().add_patch(marker)
        plt.axvspan(marker.get_x(), marker.get_x()+marker.get_width(), alpha=0.2, color="black")
        plt.gca().add_patch(marker)

    plt.xlabel('Time (ms)', fontsize=15)
    plt.ylabel('Input (Number of neurons)', fontsize=15)
    plt.show()

def plot_V_features(markers, input_data, V_t, theta=1):
    time = np.arange(0., len(input_data[0]), 1.)  # ms
    plt.rcParams['figure.figsize'] = (15.0, 7.5)
    plt.plot(time, V_t)
    m = 0
    for marker in markers:
        plt.gca().add_patch(marker)
        plt.axvspan(marker.get_x(), marker.get_x()+marker.get_width(), alpha=0.2, color="black")
        plt.gca().add_patch(marker)
        m += 1
    inp = np.argwhere(input_data == 1).T
    x = inp[1]
    y = V_t[x]
    plt.scatter(x, y, s=50, c='r')
    plt.xlabel("Time (ms)", fontsize=15)
    plt.ylabel("Voltage",  fontsize=15)
    plt.ylim((0, theta + theta/5))
    plt.axhline(y=theta, linestyle="--", color="k")
    plt.show()


def plot_raster(input_data):
    plt.rcParams['figure.figsize'] = (10, 10)
    y, x = np.argwhere(input_data > 0).T
    plt.scatter(x, y, s=35)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input (Number of neurons)')
    plt.show()


def plot_raster_weights(input_data, weights):
    plt.rcParams['figure.figsize'] = (15.0, 7.5)
    y, x = np.argwhere(input_data == 1).T
    plt.scatter(x, y, s=weights[y] * 12)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input (Number of neurons)')
    plt.show()