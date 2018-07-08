import math
import random

import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import spline, interp1d

# from pid_reg.my_pid import PID
from pid_reg.recurrent_pid import RecurrentPID
from pid_reg.standard_pid import StandardPID


def generator1(t: int):
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


def get_obj_value(x, u, t):
    h = 0
    if t < 10:
        h = -5
    elif 10 <= t < 30:
        h = -10
    elif t >= 30:
        h = 0
    return x + u + h  # + np.random.uniform(-1, 1)
    # return x + u


def generator(t):
    y = 25
    if 5 <= t < 15:
        y = 80
    elif 15 <= t < 25:
        y = 30
    elif 25 <= t < 35:
        y = 60
    return y


def get_set_point(t):
    y = 20
    if 5 <= t < 15:
        y = 70
    elif 15 <= t < 25:
        y = 70 + 2 * (t - 15)
    elif 25 <= t < 35:
        y = 90 - 1.5 * (t - 25)
    return y


# def test():
#     t = 5
#     f = 100
#     pid_reg = PID(0.9, 5000, 0.001, f)
#     # pid_reg = PID(0.3, 0.5, 0.01, f)
#     pid_reg.set_constraints(u1=-10, u2=10)
#     x = []
#     set_point = []
#     y = []
#     feedback_list = []
#     t_i = 0
#     i = 0
#     tao = 0
#     feedback=0
#
#     for i in range(t*f):
#         x.append(t_i)
#         set_point.append(generator(t_i))
#         output = pid_reg.update(set_point[i], feedback)
#         feedback += output + math.sin(t_i*10)
#         y.append(output)
#         feedback_list.append(feedback)
#         t_i += 1 / f
#
#     print(x)
#     print(set_point)
#     print(y)
#
#     # time_sm = np.array(x)
#     # time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)
#     # feedback_smooth = spline(x, y, time_smooth)
#
#     # f2 = interp1d(x, y, kind='cubic')
#
#     plt.plot(x, y)
#     plt.plot(x, set_point)
#     plt.plot(x, feedback_list)
#     plt.xlabel('time (s)')
#     plt.ylabel('PID (PV)')
#     plt.title('TEST PID')
#     plt.grid(True)
#     plt.show()


def test_recurrent_pid():
    t = 40  # время работы
    f = 5  # частота измерений, 0.2 сек.
    p = 1.
    i = 2.5
    d = 0.02
    # четыре варианта PID-регулятора.
    pid_list = [RecurrentPID(p, i, d, f),
                RecurrentPID(p, i, d, f, d_on_e=False),
                RecurrentPID(p, i, d, f, p_on_e=False),
                RecurrentPID(p, i, d, f, p_on_e=False, d_on_e=False)]

    for pid in pid_list:
        pid.set_constraints(-5, 15)

    x = []
    set_point = []
    y = [[] for _ in range(len(pid_list))]
    feedback_list = [[] for _ in range(len(pid_list))]

    t_i = 0.
    obj_values = [0 for _ in range(len(pid_list))]

    for i in range(t * f):
        x.append(t_i)
        set_point.append(generator(t_i))

        for j in range(len(pid_list)):
            feedback_list[j].append(obj_values[j])

            if i == 0:
                obj_values[j] = get_obj_value(obj_values[j], 0, t_i)
            else:
                obj_values[j] = get_obj_value(obj_values[j], y[j][-1], t_i)

            y[j].append(pid_list[j].update(set_point[i], obj_values[j]))
        t_i += 1 / f

    for i in range(len(pid_list)):
        ax = plt.subplot(221 + i)
        ax.plot(x, y[i], label="Управление")
        ax.plot(x, set_point, '--', label="Уставка")
        ax.plot(x, feedback_list[i], label="Объект")
        plt.xlabel('time (s)')
        plt.ylabel('PID (PV)')
        if i == 0:
            plt.title('Рекурентная формула')
        elif i == 1:
            plt.title('D on M')
        elif i == 2:
            plt.title('P on M')
        elif i == 3:
            plt.title('DonM and PonM')
        plt.grid(True)
        plt.legend()

    plt.show()


def test_standard_pid():
    t = 40  # время работы
    f = 5  # частота измерений, 0.2 сек.
    p = 1.
    i = 2.5
    d = 0.02
    # Четыре варианта стандартной формулы PID-регулятора.
    pid_list = [StandardPID(p, i, d, f),
                StandardPID(p, i, d, f, d_on_e=False),
                StandardPID(p, i, d, f, p_on_e=False),
                StandardPID(p, i, d, f, p_on_e=False, d_on_e=False)]

    # Установка ограничений
    for pid in pid_list:
        pid.set_constraints(-5, 15)

    t_i = 0.
    x = []
    set_point = []
    u = [[] for _ in range(len(pid_list))]  # управление
    obj_value_list = [[] for _ in range(len(pid_list))]

    for i in range(t * f):
        x.append(t_i)
        set_point.append(generator(t_i))

        for j in range(len(pid_list)):
            if i == 0:
                # Начальное значение объекта = 0
                obj_value_list[j].append(get_obj_value(0, 0, t_i))
            else:
                obj_value_list[j].append(get_obj_value(obj_value_list[j][-1], u[j][-1], t_i))

            # Расчет управления
            u[j].append(pid_list[j].update(set_point[i], obj_value_list[j][-1]))
        t_i += 1 / f

    for i in range(len(pid_list)):
        ax = plt.subplot(221 + i)
        ax.plot(x, u[i], label="Управление")
        ax.plot(x, set_point, '--', label="Уставка")
        ax.plot(x, obj_value_list[i], label="Объект")
        plt.ylabel('PID (PV)')
        if i == 0:
            plt.title('Стандартная формула')
        elif i == 1:
            plt.title('D on M')
        elif i == 2:
            plt.xlabel('time (s)')
            plt.title('P on M')
        elif i == 3:
            plt.xlabel('time (s)')
            plt.title('DonM and PonM')
        plt.grid(True)
        plt.legend()

    plt.show()


def test_pid():
    t = 40  # время работы
    f = 5  # частота измерений, 0.2 сек.
    p = 1.
    i = 2.5
    d = 0.02
    # Четыре варианта стандартной формулы PID-регулятора.
    pid_list = [RecurrentPID(p, i, d, f),
                RecurrentPID(p, i, d, f, p_on_e=False, d_on_e=False)]

    # Установка ограничений
    for pid in pid_list:
        pid.set_constraints(-5, 15)

    t_i = 0.
    x = []
    set_point = []
    u = [[] for _ in range(len(pid_list))]  # управление
    obj_value_list = [[] for _ in range(len(pid_list))]

    for i in range(t * f):
        x.append(t_i)
        set_point.append(get_set_point(t_i))

        for j in range(len(pid_list)):
            if i == 0:
                # Начальное значение объекта = 0
                obj_value_list[j].append(get_obj_value(0, 0, t_i))
            else:
                obj_value_list[j].append(get_obj_value(obj_value_list[j][-1], u[j][-1], t_i))

            # Расчет управления
            u[j].append(pid_list[j].update(set_point[i], obj_value_list[j][-1]))
        t_i += 1 / f

    for i in range(len(pid_list)):
        ax = plt.subplot(121 + i)
        ax.plot(x, u[i], label="Управление")
        ax.plot(x, set_point, '--', label="Уставка")
        ax.plot(x, obj_value_list[i], label="Объект")
        plt.ylabel('PID (PV)')
        plt.xlabel('time (s)')
        if i == 0:
            plt.title('Стандартная формула')
        elif i == 1:
            plt.title('D on M')
        plt.grid(True)
        plt.legend()

    plt.show()


def main():
    # test()

    # Тест реализации стандартной формулы PID-регулятора
    test_standard_pid()

    test_recurrent_pid()

    test_pid()

if __name__ == "__main__":
    main()
