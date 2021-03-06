import numpy as np
import matplotlib.pyplot as plt

from identification.adaptive_lsm import lsm_linear_parametrization
from identification.identifier import Identifier
from model.model_obj_builder import create_model
from pid_reg.my_pid import PID


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


def test():
    t = 40
    f = 5  # 0.2 сек.

    # pid_reg = PID(0.8, 0.15, 0.1, f)
    pid = PID(1, 0.5, 0.1, f)
    pid.set_constraints(u1=-5, u2=15)

    x = []
    set_point = []
    y = []
    feedback_list = []

    t_i = 0.
    feedback = 0

    for i in range(t*f):
        x.append(t_i)
        set_point.append(generator(t_i))

        feedback_list.append(feedback)
        if i == 0:
            feedback = get_obj_value(feedback, 0, t_i)
        else:
            feedback = get_obj_value(feedback, y[-1], t_i)

        output = pid.update(set_point[i], feedback)

        y.append(output)
        t_i += 1 / f
    print(x)
    print(set_point)
    print(y)
    print(feedback_list)

    # time_sm = np.array(x)
    # time_smooth = np.linspace(time_sm.min(), time_sm.max(), num=1001, endpoint=True)
    # feedback_smooth = spline(x, feedback_list, time_smooth)

    # f2 = interp1d(x, feedback_list, kind='cubic')

    # plt.plot(time_smooth, feedback_smooth, label="2")
    # plt.plot(time_smooth, f2(time_smooth), label="1")

    plt.plot(x, y, label="Управление")
    plt.plot(x, set_point, '--', label="Уставка")
    plt.plot(x, feedback_list, label="Объект")
    plt.xlabel('time (s)')
    plt.ylabel('PID (PV)')
    plt.title('TEST PID')

    plt.grid(True)
    plt.legend()
    plt.show()


def test_asi():
    model = create_model("x(t-1)+u(t-1)+a0")
    print(model.model_expr)
    print(model.obj_expr)

    print(model.x_names)
    print(model.u_names)
    print(model.a_names)

    model.initialization(a0=0)
    print(model.value_x)
    print(model.value_u)
    print(model.value_a)

    f = 5
    t = 40
    t_i = 0.
    x = []
    set_point = []  # уставка
    obj_values = []  # значения объекта
    u_val = []  # значенея управления
    y = []
    a_val = np.zeros(f * t)  # значния коэффициента
    obj = 0
    u = 0
    model.number_averaged_values = 1
    identifier = Identifier(model=model, n=1, sigma=0.1)
    grad = []
    for i in range(t*f):
        x.append(t_i)
        set_point.append(generator(t_i))
        u_val.append(u)

        obj_values.append(obj)
        if i == 0:
            obj = get_obj_value(obj, 0, t_i)
        else:
            obj = get_obj_value(obj, u_val[-1], t_i)

        if i < identifier.n_0:
            y.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
            grad.append([func(*model.value_x, *model.value_u, *model.value_a) for func in model.grad])
            model.value_a = [a_val[i]]
            a_val[i + 1] = a_val[i]
        else:
            # Начинаем проводить корректировку параметров модели.
            if i == identifier.n_0:
                identifier.grad = grad
            y.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
            a_n = lsm_linear_parametrization(identifier, obj, 0.33, lbd=0.5)
            model.value_a = list(a_n)
            a_val[i] = a_n

        model.value_x = [obj]

        v = generator(t_i + 1 / f) - obj - a_val[i]

        if v < -10:
            v = -10
        elif v > 20:
            v = 20

        model.value_u = [v]
        u = v

        t_i += 1 / f

    print(u_val)
    print(y)
    print(obj_values)
    print(a_val)

    plt.plot(x, y, label="Модель")
    plt.plot(x, u_val, label="Управление")
    plt.plot(x, obj_values, label="Объект")
    plt.plot(x, set_point, '--', label="Уставка")
    plt.plot(x, a_val, label="a0")
    plt.grid(True)
    legend = plt.legend()
    legend.draggable(True)
    plt.show()


def main():
    # test()
    test_asi()
    # pid_reg = PID(1, 0.5, 0.1, 10)
    # print(pid_reg)

if __name__ == "__main__":
    main()
