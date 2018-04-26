from sympy.utilities.autowrap import ufuncify
import sympy as sp

from model.parser import parse_obj_expr


class Model:
    def __init__(self, obj_expr: str):
        self.obj_expr_str = obj_expr  # уравнение объекта, в виде строки

        self._model_expr_str = ""
        self._a_names_str = []
        self._u_names_str = []
        self._x_names_str = []
        self._coefficient_map = {}

        self._tao = []
        self._tao_u = []
        self._tao_x = []

        self._obj_expr = None
        self._model_expr = None  # уравнение модель sumpy
        self._a_names = None
        self._u_names = None
        self._x_names = None

        self._value_a = []
        self._value_u = []
        self._value_x = []

        self._grad = None

        self.func_value_obj = None
        self.find_u_func = None
        self.func_model = None

        self.f = None

        # применение сглаживания
        self._last_a = []  # k предыдущих значений каждого коэффициента.
        self._number_averaged_values = 1
        self._last_delta = 0.

    def initialization(self, *args, **kwargs):
        self._value_a = [.0 for _ in self._a_names]
        self._value_u = [.0 for _ in self._u_names]
        self._value_x = [.0 for _ in self._x_names]
        if args != ():
            i = 0
            for a in args:
                if i < len(self._value_x):
                    self._value_x[i] = a
                elif len(self._value_x) <= i < len(self._value_x) + len(self._value_u):
                    self._value_u[i - len(self._value_x)] = a
                elif len(self._value_x) + len(self._value_u) <= i < len(self._value_x) + len(self._value_u) + len(self._value_a):
                    self._value_a[i - len(self._value_x) - len(self._value_u)] = a
                i += 1
        if kwargs != {}:
            for a in kwargs.keys():
                if a in self._x_names_str:
                    self._value_x[self._x_names_str.index(a)] = kwargs.get(a)
                elif a in self._u_names_str:
                    self._value_u[self._u_names_str.index(a)] = kwargs.get(a)
                elif a in self._a_names_str:
                    self._value_a[self._a_names_str.index(a)] = kwargs.get(a)

    def find_value_model(self):
        pass

    @property
    def model_expr_str(self):
        return self._model_expr_str

    @model_expr_str.setter
    def model_expr_str(self, value):
        self._model_expr_str = value

    @property
    def a_names_str(self):
        return self._a_names_str

    @a_names_str.setter
    def a_names_str(self, value):
        self._a_names_str = value

    @property
    def u_names_str(self):
        return self._u_names_str

    @u_names_str.setter
    def u_names_str(self, value):
        self._u_names_str = value

    @property
    def x_names_str(self):
        return self._x_names_str

    @x_names_str.setter
    def x_names_str(self, value):
        self._x_names_str = value

    @property
    def tao_u(self):
        return self._tao_u

    @tao_u.setter
    def tao_u(self, value):
        self._tao_u = value

    @property
    def tao_x(self):
        return self._tao_x

    @tao_x.setter
    def tao_x(self, value):
        self._tao_x = value

    @property
    def obj_expr(self):
        return self._obj_expr

    @obj_expr.setter
    def obj_expr(self, value):
        self._obj_expr = value

    @property
    def model_expr(self):
        return self._model_expr

    @model_expr.setter
    def model_expr(self, value):
        self._model_expr = value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def a_names(self):
        return self._a_names

    @a_names.setter
    def a_names(self, value):
        self._a_names = value

    @property
    def u_names(self):
        return self._u_names

    @u_names.setter
    def u_names(self, value):
        self._u_names = value

    @property
    def x_names(self):
        return self._x_names

    @x_names.setter
    def x_names(self, value):
        self._x_names = value

    @property
    def value_a(self):
        return self._value_a

    @value_a.setter
    def value_a(self, value):
        if type(value) is list and len(value) == len(self._a_names):
            if len(self._last_a) != 0 and len(self._last_a) == self._number_averaged_values:
                self._last_a.pop(0)
            if len(self._value_a) != 0:
                self._last_a.append(self._value_a)
            # for i in range(len(self._value_a)):
            #     if len(self._last_a) != 0 and len(self._last_a) == self._number_averaged_values:
            #         self._last_a.pop(0)
            #     self._last_a.append(value)
            self._value_a = value

    @property
    def value_u(self):
        return self._value_u

    @value_u.setter
    def value_u(self, value):
        if type(value) is list and len(value) == len(self._u_names):
            self._value_u = value

    @property
    def value_x(self):
        return self._value_x

    @value_x.setter
    def value_x(self, value):
        if type(value) is list and len(value) == len(self._x_names):
            self._value_x = value

    @property
    def last_a(self):
        return self._last_a

    @last_a.setter
    def last_a(self, value, idx=0):
        if idx < len(self._last_a):
            if len(self._last_a[idx]) != 0:
                self._last_a[idx].pop(0)
            self._last_a[idx].append(value)

    @property
    def last_delta(self):
        return self._last_delta

    @last_delta.setter
    def last_delta(self, value):
        self._last_delta = value

    @property
    def number_averaged_values(self):
        return self._number_averaged_values

    @number_averaged_values.setter
    def number_averaged_values(self, value: int):
        self._number_averaged_values = value


def find_grad(expr, coefficients, params):
    grad = []
    for c in coefficients:
        print(expr.diff(c))
        derivative = ufuncify(params, expr.diff(c))
        grad.append(derivative)
    return grad


def create_model(obj_expr: str, delimiter=""):
    model, obj, x_names, u_names, a_names, tao_x, tao_u = parse_obj_expr(obj_expr, delimiter=delimiter)

    m = Model(obj_expr)
    m.model_expr_str = model
    m.a_names_str = a_names
    m.u_names_str = u_names
    m.x_names_str = x_names
    m.tao_x = tao_x
    m.tao_u = tao_u

    m.model_expr = sp.S(model)
    m.obj_expr = sp.S(obj)

    # sp.var(<list[str]>) -> list[variables]
    m.x_names = sp.var(m.x_names_str)
    m.u_names = sp.var(m.u_names_str)
    m.a_names = sp.var(m.a_names_str)

    # m.x_names, m.u_names, m.a_names
    # m.grad(пераметры через запятую) или m.grad(*<список параметров>)
    m.grad = find_grad(m.model_expr, m.a_names, m.x_names + m.u_names + m.a_names)

    m.func_value_obj = ufuncify(m.x_names + m.u_names + m.a_names, m.obj_expr)
    m.func_model = ufuncify(m.x_names + m.u_names + m.a_names, m.model_expr)

    return m


def main():
    model = create_model("a_0+a_1*x(t-1)+a_2*x(t-2)")
    print(model.model_expr)
    print(model.obj_expr)

    print(model.x_names)
    print(model.u_names)
    print(model.a_names)

    model.number_averaged_values = 5

    model.value_a = [1, 1, 1]
    print(model.value_a)
    print(model.last_a)
    model.value_a = [2, 2, 2]
    print(model.value_a)
    print(model.last_a)
    model.value_a = [3, 3, 3]
    print(model.value_a)
    print(model.last_a)
    model.value_a = [4, 4, 4]
    print(model.value_a)
    print(model.last_a)
    model.value_a = [5, 5, 5]
    print(model.value_a)
    print(model.last_a)
    model.value_a = [6, 6, 6]
    print(model.value_a)
    print(model.last_a)


if __name__ == "__main__":
    main()

