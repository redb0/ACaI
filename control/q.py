import random

import sympy as sp
from sympy.utilities.autowrap import ufuncify

from identification.simplest_adaptive_algorithm import simplest_adaptive_algorithm
from model.model_obj_builder import create_model


import matplotlib.pyplot as plt
import math


def control(model, set_point):
    # alpha = 1
    idx_alpha = 0
    for u in model.u_names_str:
        if u == "u0":
            alpha_str = str(sp.diff(model.model_expr, u))
            idx_alpha = model.a_names_str.index(alpha_str)
            # alpha = model.value_a(idx_alpha)
            break
    set_point_var = sp.Symbol("sp")
    v_expr = (set_point_var - model.model_expr + sp.diff(model.model_expr, model.a_names[idx_alpha]) * model.a_names[idx_alpha]) / model.a_names[idx_alpha]
    # print(v_expr)
    if model.f is None:
        model.f = ufuncify([set_point_var] + model.x_names + model.u_names + model.a_names, v_expr)
    # f = ufuncify([set_point_var] + model.x_names + model.u_names + model.a_names, v_expr)
    return set_point_var, v_expr


def g(t):
    # if t < 0.3:
    #     return 0
    if t < 5:
        return 80
    elif 5 <= t < 10:
        return 40
    elif 10 <= t < 20:
        return 30
    else:
        return 60


def main():
    model = create_model("a_0+a_1*x(t-1)+a_2*u(t-1)")
    print(model.model_expr)
    print(model.obj_expr)

    print(model.x_names)
    print(model.u_names)
    print(model.a_names)

    model.initialization(a0=1, a1=1, a2=0.5)
    print(model.value_x)
    print(model.value_u)
    print(model.value_a)

    u = [0]
    y = [0]
    a0 = [0.1]
    a1 = [0.1]
    a2 = [0.1]
    t_a = [0]
    obj = [0]
    setp = [0.]

    model.value_u = u
    model.value_x = obj

    t = 0.
    d_t = 0.25
    u_t = 0
    output_object = 0
    v = 0
    i = 0
    model.number_averaged_values = 5
    while t < 30:
        t += d_t
        i += 1
        set_point = g(t)
        setp.append(set_point)
        output_object = -5 + 0.5 * obj[-1] + u[-1]  # + math.sin(10*t) + random.uniform(-3, 3)
        # model.value_x = [output_object]

        # model.value_x = [output_object]
        # if t < 6:
        # y1, new_a = simplest_adaptive_algorithm(model, output_object)
        if i < 6:
            y1, new_a = simplest_adaptive_algorithm(model, output_object, smoothing=0, n=i, l=0.9)
        else:
            y1, new_a = simplest_adaptive_algorithm(model, output_object, smoothing=1, n=i, l=0.9)
        model.value_a = new_a

        # y1 = model.func_model(*model.value_x, *model.value_u, *model.value_a)

        model.value_x = [output_object]

        # if t >= 0.2:
        set_point_var, v_expr = control(model, set_point)

        v = model.f(set_point, *model.value_x, *model.value_u, *model.value_a)
        # else:
        #     v = 1
        # if v < 0:
        #     v = 0
        # elif v > 100:
        #     v = 100

        if v < 0:
            v = 0
        elif v > 100:
            v = 100

        model.value_u = [v]
        u.append(v)

        a0.append(new_a[0])
        a1.append(new_a[1])
        a2.append(new_a[2])

        y.append(y1)

        # u.append(v)
        obj.append(output_object)
        t_a.append(t)

    print(v_expr)

    # print(t_a)
    print(u)
    print(y)
    print(obj)
    print(a0)
    print(a1)
    print(a2)

    plt.plot(t_a, y, label="Модель")
    plt.plot(t_a, u, label="Управление")
    plt.plot(t_a, obj, label="Объект")
    plt.plot(t_a, setp, label="Уставка")
    plt.plot(t_a, a0, label="a0")
    plt.plot(t_a, a1, label="a1")
    plt.plot(t_a, a2, label="a2")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
