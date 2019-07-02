import pyautogui
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import time
import matplotlib.animation as animation
import os


def heatmap_mouse_event(w, h):
    screen_h = h
    screen_w = w

    data = np.zeros((screen_w, screen_h))

    data_history = []
    data_history.append(data)
    i = 0

    while i < 500:
        time.sleep(0.001)
        print("Time :{}".format(i))
        pos = pyautogui.position()
        address = (pos[0], pos[1])
        print(address)

        if address[0] < w and address[1] < h:
            data = np.zeros((screen_w, screen_h))
            data[address] = 1
            data_history.append(data)

            t_start = time.time()

            plt.imshow(data, animated=True)
            t_end = time.time()
            print("fps = {0}".format(999 if t_end == t_start else 1/(t_end-t_start)))
            i += 1

        else:
            print("Outside of recoding area")

    return data


def main(w, h):
    print("Main")

    fig = plt.figure()
    data = heatmap_mouse_event(w,h)
    anim = animation.ArtistAnimation(fig, data)

    # anim.save('dynamic_images.mp4')
    # show plot
    plt.show()


if __name__ == '__main__':
    import argparse

    text_type = str

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-x', '--x',
        default=150,
        type=int,
    )

    parser.add_argument(
        '-y', '--y',
        default=150,
        type=int,
    )

    args, unknown = parser.parse_known_args()
    try:
        main(
            args.x,
            args.y,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
