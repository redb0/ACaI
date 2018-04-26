import numpy as np
import matplotlib.pyplot as plt

from identification.adaptive_lsm import lsm_linear_parametrization, lsm_linear_parametrization_efi
from identification.identifier import Identifier
from model.model_obj_builder import create_model


def generate_signal(x, x1, ideal=True):
    if ideal:
        return 5 + 0.8 * x - 3 * np.sin(x1)
    else:
        return 5 + 0.8 * x - 3 * np.sin(x1) + 0.5 * np.cos(x) + np.random.uniform(-1, 1)


def example_1():
    # Создаем линейную модель
    model = create_model("a_0+a_1*x(t-1)+a_2*sin(x(t-2))")
    # Инициализируем коэффициенты нулями, либо заранее известными присблизительными значениями
    model.initialization(a0=0, a1=0, a2=0)

    # Создаем идентификатор. Будем накапливать 5 измерений,
    # измерения - равноточные (среднеквадратичное отклонение одинаково).
    identifier = Identifier(model=model, n=5, sigma=0.3**2)

    # Имитируем 50 тактов
    N = 50

    obj = [0.0]
    a = np.zeros((N + 1, 3))
    a[0] = np.array(model.value_a)
    t_a = [0]
    model_values = [0.]

    # Инициализируем первые значения объекта нулями.
    model.value_x = [.0, .0]

    t = 0.
    d_t = 0.1

    # контейнер значений градиентов накопленных измерений
    grad = []
    for i in range(N):
        # измеряем выход объекта в случае полной информации об объекте (нет помех и неизвестных воздействий)
        if i == 0:
            obj_value = generate_signal(obj[-1], 0, ideal=True)
        else:
            obj_value = generate_signal(obj[-1], obj[-2], ideal=True)

        # Накапливаем измерения, и сразу вычисляем градиенты.
        if i < identifier.n_0:
            grad.append([func(*model.value_x, *model.value_u, *model.value_a) for func in model.grad])
            model.value_a = list(a[-1])
            a[i + 1] = a[i]
        else:
            # Начинаем проводить корректировку параметров модели.
            if i == identifier.n_0:
                identifier.grad = grad
            a_n = lsm_linear_parametrization(identifier, obj_value, 0.3**2, lbd=1.)
            model.value_a = list(a_n)
            a[i + 1] = a_n

        # Рассчитываем значения модели
        model_values.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
        # Обновление значения выхода объекта в модели.
        if i == 0:
            model.value_x = [obj_value, .0]
        else:
            model.value_x = [obj_value, obj[-1]]
        obj.append(obj_value)
        t += d_t
        t_a.append(t)

    print(obj)
    print(model_values)

    plt.plot(t_a, obj, label='Объект')
    plt.plot(t_a, model_values, label='Модель')
    plt.plot(t_a, a[:, 0], label='Параметр a0')
    plt.plot(t_a, a[:, 1], label='Параметр a1')
    plt.plot(t_a, a[:, 2], label='Параметр a2')

    plt.xlabel("Время, сек")
    plt.title("Идентификация линейной модели классическим алгоритмом МНК")

    plt.grid()
    plt.legend()
    plt.show()


def example_2():
    # Создаем линейную модель
    model = create_model("a_0+a_1*x(t-1)+a_2*sin(x(t-2))")
    model.initialization(a0=1, a1=0.1, a2=0.2)

    # Создаем идентификатор. Будем накапливать 5 измерений,
    # измерения - неравноточные (среднеквадратичное отклонение разное).
    identifier = Identifier(model=model, n=5, sigma=[np.random.uniform(0, 0.5)**2 for _ in range(5)])

    N = 50

    obj = [0.0]
    a = np.zeros((N + 1, 3))
    a[0] = np.array(model.value_a)
    t_a = [0]
    model_values = [0.]

    model.value_x = [.0, .0]

    t = 0.
    d_t = 0.1

    grad = []
    for i in range(N):
        # измеряем выход объекта в случае неполной информации об объекте (есть неизвестная составляющая и помеха)
        if i == 0:
            obj_value = generate_signal(obj[-1], 0, ideal=False)
        else:
            obj_value = generate_signal(obj[-1], obj[-2], ideal=False)

        # Накапливаем измерения, и сразу вычисляем градиенты.
        if i < identifier.n_0:
            grad.append([func(*model.value_x, *model.value_u, *model.value_a) for func in model.grad])
            model.value_a = list(a[i])
            a[i + 1] = a[i]
        else:
            # Начинаем проводить корректировку параметров модели.
            if i == identifier.n_0:
                identifier.grad = grad
            a_n = lsm_linear_parametrization(identifier, obj_value, np.random.uniform(0, 0.5)**2, lbd=0.9)
            model.value_a = list(a_n)
            a[i + 1] = a_n

        # Рассчитываем значения модели
        model_values.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
        # Обновление значения выхода объекта в модели.
        if i == 0:
            model.value_x = [obj_value, .0]
        else:
            model.value_x = [obj_value, obj[-1]]
        obj.append(obj_value)
        t += d_t
        t_a.append(t)

    print(obj)
    print(model_values)

    plt.plot(t_a, obj, label='Объект')
    plt.plot(t_a, model_values, label='Модель')
    plt.plot(t_a, a[:, 0], label='Параметр a0')
    plt.plot(t_a, a[:, 1], label='Параметр a1')
    plt.plot(t_a, a[:, 2], label='Параметр a2')

    plt.xlabel("Время, сек")
    plt.title("Идентификация линейной модели классическим алгоритмом МНК")

    plt.grid()
    plt.legend()
    plt.show()


def example_3():
    # Создаем линейную модель
    model = create_model("a_0+a_1*x(t-1)+a_2*sin(x(t-2))")
    model.initialization(a0=1, a1=0.1, a2=0.2)

    # Создаем идентификатор. Будем накапливать 5 измерений,
    # измерения - неравноточные (среднеквадратичное отклонение разное).
    identifier = Identifier(model=model, n=10, sigma=0.01)

    N = 50

    obj = [0.0]
    a = np.zeros((N + 1, 3))
    a[0] = np.array(model.value_a)
    t_a = [0]
    model_values = [0.]

    model.value_x = [.0, .0]

    t = 0.
    d_t = 0.1

    grad = []
    for i in range(N):
        # измеряем выход объекта в случае неполной информации об объекте (есть неизвестная составляющая и помеха)
        if i == 0:
            obj_value = generate_signal(obj[-1], 0, ideal=True)
        else:
            obj_value = generate_signal(obj[-1], obj[-2], ideal=True)

        # Накапливаем измерения, и сразу вычисляем градиенты.
        if i < identifier.n_0:
            grad.append([func(*model.value_x, *model.value_u, *model.value_a) for func in model.grad])
            model.value_a = list(a[i])
            a[i + 1] = a[i]
        else:
            # Начинаем проводить корректировку параметров модели.
            if i == identifier.n_0:
                identifier.grad = grad
            a_n = lsm_linear_parametrization(identifier, obj_value, 0.01, lbd=0.9)
            model.value_a = list(a_n)
            a[i + 1] = a_n

        # Рассчитываем значения модели
        model_values.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
        # Обновление значения выхода объекта в модели.
        if i == 0:
            model.value_x = [obj_value, .0]
        else:
            model.value_x = [obj_value, obj[-1]]
        obj.append(obj_value)
        t += d_t
        t_a.append(t)

    print(obj)
    print(model_values)
    print("a0 = ", a[:, 0][-1])
    print("a1 = ", a[:, 1][-1])
    print("a2 = ", a[:, 2][-1])

    plt.plot(t_a, obj, label='Объект')
    plt.plot(t_a, model_values, label='Модель')
    plt.plot(t_a, a[:, 0], label='Параметр a0')
    plt.plot(t_a, a[:, 1], label='Параметр a1')
    plt.plot(t_a, a[:, 2], label='Параметр a2')

    plt.xlabel("Время, сек")
    plt.title("Идентификация линейной модели классическим алгоритмом МНК")

    plt.grid()
    plt.legend()
    plt.show()


def example_4():
    # Создаем линейную модель
    model = create_model("a_0+a_1*sin(a_2*x(t-1))")
    # model = create_model("a_0+a_1*(x(t-1)**a_2)")
    model.initialization(a0=1, a1=1, a2=1)

    # Создаем идентификатор. Будем накапливать 5 измерений,
    # измерения - неравноточные (среднеквадратичное отклонение разное).
    identifier = Identifier(model=model, n=5, sigma=0.1)

    N = 200

    obj = [1.]
    a = np.zeros((N + 1, 3))
    a[0] = np.array(model.value_a)
    t_a = [0]
    model_values = [0.]

    model.value_x = [1.]

    t = 0.
    d_t = 0.1

    grad = []
    for i in range(N):
        # измеряем выход объекта в случае неполной информации об объекте (есть неизвестная составляющая и помеха)
        obj_value = 5 + 2 * np.sin(0.7 * obj[-1]) + np.random.uniform(-0.1, 0.1)
        # obj_value = 5 + 2 * (obj[-1]**(-2))  # + np.random.uniform(-0.1, 0.1)

        # Накапливаем измерения, и сразу вычисляем градиенты.
        if i < identifier.n_0:
            grad.append([func(*model.value_x, *model.value_u, *model.value_a) for func in model.grad])
            model.value_a = list(a[i])
            a[i + 1] = a[i]
        else:
            # Начинаем проводить корректировку параметров модели.
            if i == identifier.n_0:
                identifier.grad = grad
            a_n = lsm_linear_parametrization(identifier, obj_value, 0.1, lbd=0.95)
            model.value_a = list(a_n)
            a[i + 1] = a_n

        # Рассчитываем значения модели
        model_values.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
        # Обновление значения выхода объекта в модели.
        model.value_x = [obj_value]
        obj.append(obj_value)
        t += d_t
        t_a.append(t)

    print(obj)
    print(model_values)

    print(a[:, 0])
    print(a[:, 1])
    print(a[:, 2])

    print("a0 = ", a[:, 0][-1])
    print("a1 = ", a[:, 1][-1])
    print("a2 = ", a[:, 2][-1])

    plt.plot(t_a, obj, label='Объект')
    plt.plot(t_a, model_values, label='Модель')
    plt.plot(t_a, a[:, 0], label='Параметр a0')
    plt.plot(t_a, a[:, 1], label='Параметр a1')
    plt.plot(t_a, a[:, 2], label='Параметр a2')

    plt.xlabel("Время, сек")
    plt.title("Идентификация линейной модели классическим алгоритмом МНК")

    plt.grid()
    plt.legend()
    plt.show()


def main():
    # example_1()
    # example_2()
    # example_3()
    example_4()

if __name__ == "__main__":
    main()
