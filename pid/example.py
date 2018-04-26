import math
import random

import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import spline, interp1d

from pid.my_pid import PID


def generator(t: int):
    # return math.sin(t) + math.cos(6*t) + 1
    # if t < 5:
    #     return 5
    # elif 5 <= t < 10:
    #     return 50
    # elif 10 <= t < 15:
    #     return 20
    y = 5 + 3 * math.sin(5 * t) + 2 * math.cos(10 * t) + 5 * math.sin(20 * t) + random.uniform(0, 1)
    return y
    # return 4


def test():
    t = 5
    f = 100
    pid = PID(0.9, 5000, 0.001, f)
    pid.set_constraints(u1=-10, u2=10)

    x = []
    set_point = []
    y = []
    feedback_list = []

    t_i = 0
    i = 0
    tao = 0
    feedback=0

    for i in range(t*f):
        x.append(t_i)
        set_point.append(generator(t_i))
        output = pid.update(set_point[i], feedback)
        feedback += output + math.sin(t_i*10)
        y.append(output)
        # if i < tao:
        #     y.append(0)
        #     feedback += y[i]
        # else:
        #     output = pid.update(set_point[i], feedback)
        #     y.append(output)
        #     feedback += y[i - 1 - tao]
        feedback_list.append(feedback)
        t_i += 1 / f



    # while t_i < t:
    #     x.append(t_i)
    #     set_point.append(generator(t_i))
    #     if i == 0:
    #         y.append(pid.update(0.0, 0.))
    #     else:
    #         # if i > tao:
    #         output = pid.update(set_point[i], feedback)
    #         feedback += output
    #         feedback_list.append(feedback)
    #         y.append(output)
    #         # else:
    #         #     y.append(0)

        # t_i += 1 / f
        # i += 1
    print(x)
    print(set_point)
    print(y)

    # time_sm = np.array(x)
    # time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)
    # feedback_smooth = spline(x, y, time_smooth)

    # f2 = interp1d(x, y, kind='cubic')

    plt.plot(x, y)
    plt.plot(x, set_point)
    plt.plot(x, feedback_list)
    plt.xlabel('time (s)')
    plt.ylabel('PID (PV)')
    plt.title('TEST PID')

    plt.grid(True)
    plt.show()


def main():
    test()

if __name__ == "__main__":
    main()
