import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class AdaptiveControlAlgorithm:
    def __init__(self, obj_model_expr, u1=0, u2=100, id_alg_idx=1):
        self.obj_model_expr = obj_model_expr  # пользователь задает уравнение объекта
        # пользователь задает ограничение на управление
        self.u1 = u1  # ограничение снизу
        self.u2 = u2  # ограничения сверху

        self.identification_algorithm = id_alg_idx  # индекс алгоритма идентификации

        self.obj_expr = None  # sympy выражение для уравнения объекта
        self.model_expr = None  # sympy выражение для модели объекта

        self.coefficients = []  # массив коэффициентов "альфа"

        self.perfect_control = None  # sympy выражение для идеального управления

        self.identification_error = None  # error 1, ошибка идентификации
        self.error_limited_control = None  # error 2, ошибка связанная с накладываемыми на управление ограничениями

    def get_control_action(self):
        pass

    @property
    def obj_model_expr(self):
        return self.obj_model_expr

    @obj_model_expr.setter
    def obj_model_expr(self, value):
        self.obj_model_expr = value

    @property
    def u1(self):
        return self.u1

    @u1.setter
    def u1(self, value):
        self.u1 = value

    @property
    def u2(self):
        return self.u2

    @u2.setter
    def u2(self, value):
        self.u2 = value


def generator(t):
    # return math.sin(t) + math.cos(6*t) + 1
    if t < 5:
        return 5
    elif 5 <= t < 10:
        return 50
    elif 10 <= t < 15:
        return 20
    else:
        return 10 + np.math.sin(5 * t)
    # y = 5 + 3 * np.math.sin(5 * t)
    # return y


def obj(u, x):
    y = x + u + np.random.uniform(-1, 1)
    return y


def model(x, u, a):
    y = x + u + a
    return y


def get_a(x, x_1, u):
    return x - x_1 - u


def test():
    d_t = 0.1
    t = 0
    t_s = 35
    u1 = -100
    u2 = 100

    x = []
    u = []
    y = []
    s = []
    c = []

    idx = 0
    fedback = 0
    while t < t_s:
        set_point = generator(t + d_t)

        if t == 0:
            # set_point = 0
            x_t = 0
            v = 0
            coef_a = 0
            # c.append(coef_a)
        else:
            # set_point = generator(t)
            x_t = y[idx - 1] + u[idx - 1] + np.random.uniform(-0.5, 0.5)
            coef_a = x_t - y[idx - 1] - u[idx - 1]
            v = set_point - x_t - coef_a
            if v < u1:
                v = u1
            elif v > u2:
                v = u2

        c.append(coef_a)
        # y_model = model(x_t, v, a)
        u.append(v)
        y.append(x_t)
        s.append(set_point)
        x.append(t)
        t = t + d_t
        idx += 1
    print(x)
    print(u)
    print(s)
    print(y)
    print(c)

    plt.plot(x, u, label="Управление")
    plt.plot(x, s, label="Уставка")
    plt.plot(x, y, label="Объект")
    plt.plot(x, c, label="Коэффициент")
    plt.legend()
    plt.grid(True)
    plt.show()


def adaptive_alg(expr_obj, expr_model, s):
    pass


def main():
    test()

if __name__ == "__main__":
    main()
