import random

import numpy as np

from model.model_obj_builder import create_model

import matplotlib.pyplot as plt


def simplest_adaptive_algorithm(model, output_object, smoothing=0, n=1, l=0.9):
    new_a = []
    grad = []
    grad_func = model.grad
    for i in range(len(grad_func)):
        grad.append(grad_func[i](*model.value_x, *model.value_u, *model.value_a))

    # y = model.func_value_obj(*model.value_x, *model.value_u, *model.value_a)
    y = model.func_model(*model.value_x, *model.value_u, *model.value_a)

    delta = (output_object - y) / (np.sum([i**2 for i in grad]))

    for i in range(len(grad_func)):
        new_a.append(model.value_a[i] + delta * grad[i])

    # if smoothing == 0:
    #     model.value_a = new_a
    if smoothing == 1:
        new_a, delta_n = smoothing_efi(new_a, model.last_a, model.last_delta, l=l)
        model.last_delta = delta_n
    elif smoothing == 2:
        new_a = saa_moving_average_method(new_a, model.last_a, model.number_averaged_values)
    elif smoothing == 3:
        new_a = simple_smoothing(new_a, model.last_a, n)

    model.value_a = new_a

    return y, new_a


def simple_smoothing(a_n, last_a, n):
    for i in range(len(a_n)):
        a_n[i] = last_a[-1][i] + (a_n[i] - last_a[-1][i]) / n
    return a_n


def smoothing_efi(a_n, last_a, last_delta, l=0.9):
    if l <= 0 or l >= 1:
        print("Недопустимое значение параметра l, корректные значения: 0 < l < 1")
        return None
    delta_n = 1 + l * last_delta

    for i in range(len(a_n)):
        a_n[i] = last_a[-1][i] + (a_n[i] - last_a[-1][i]) / delta_n

    return a_n, delta_n


def saa_moving_average_method(a_n, last_a, k):
    for i in range(len(a_n)):
        if len(last_a) == k:
            a_n[i] = last_a[-1][i] + (a_n[i] - last_a[0][i]) / k
        else:
            print("Количество прошлых значений коэффициента " + str(i) + " должно быть равно " + str(k))
            return None
    return a_n


def g(t):
    if t < 5:
        return 5
    elif 5 <= t < 10:
        return 50
    elif 10 <= t < 15:
        return 20
    else:
        return 10 + np.math.sin(5 * t)


def g1(t, u):
    return 5 + 2 * t + 3 * np.math.cos(u) + random.uniform(-0.15, 0.15)


def main():
    model = create_model("a_0+a_1*x(t-1)+a_2*u(t-1)")
    print(model.model_expr)
    print(model.obj_expr)

    print(model.x_names)
    print(model.u_names)
    print(model.a_names)

    model.initialization()
    print(model.value_x)
    print(model.value_u)
    print(model.value_a)
    import sympy as sp
    # print("Производная по u0 ---> ", sp.diff(model.model_expr, model.u_names[0]))
    # print(model.model_expr - sp.diff(model.model_expr, model.u_names[0])*model.u_names[0])

    # model.initialization(1, 2)
    # print(model.value_x)
    # print(model.value_u)
    # print(model.value_a)
    #
    # model.initialization(3, 4, a1=5, a2=10)
    # print(model.value_x)
    # print(model.value_u)
    # print(model.value_a)

    u = []
    y = []
    a0 = []
    a1 = []
    a2 = []
    t_a = []
    obj = []

    t = 0
    d_t = 0.1
    u_t = 0
    i = 0
    while t < 30:
        i += 1
        output_object = g1(t, u_t)
        # model.value_x = [output_object, output_object]
        u_t = g(t)
        if i < 2:
            y1, new_a = simplest_adaptive_algorithm(model, output_object, smoothing=0, n=i, l=0.9)
        else:
            y1, new_a = simplest_adaptive_algorithm(model, output_object, smoothing=0, n=i, l=0.9)
        model.value_x = [output_object]
        a0.append(new_a[0])
        a1.append(new_a[1])
        a2.append(new_a[2])
        y.append(y1)

        u.append(u_t)
        obj.append(output_object)
        t_a.append(t)
        t += d_t

    print(t_a)
    # print(u)
    print(y)
    print(obj)
    print(a0)
    print(a1)
    print(a2)

    plt.plot(t_a, y, label="y")
    # plt.plot(t_a, u, label="u")
    plt.plot(t_a, obj, label="obj")
    plt.plot(t_a, a0, label="a0")
    plt.plot(t_a, a1, label="a1")
    plt.plot(t_a, a2, label="a2")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
