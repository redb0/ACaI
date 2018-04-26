import numpy as np

import matplotlib.pyplot as plt

from model.model_obj_builder import create_model
from identification.identifier import Identifier


def get_gamma(k, grad, w):
    """
    Рассчет коэффициента g_n.
    
    :param k    : матрица (корелляционная или Г_a).
    :param grad : вектор-столбец градиента (в нелинейном случае) 
                  или базисных функций.
    :param w    : весосвой коэффициент.
    :return: вектор столбец коэффициентов gamma, длина равна количеству параметров a.
    """
    gamma = (k @ grad) / (w + grad.T @ k @ grad)
    return gamma


def get_a(last_a, gamma, v_obj, model_value):
    """
    Расчет новых коэффициентов альфа.
    :param last_a      : коэффициенты альфа на предыдущем шаге (вектор-столбец).
    :param gamma       : коэффициенты гамма (вектор-столбец).
    :param v_obj       : измерение выхода объекта.
    :param model_value : значение модели при текущих значениях входа 
                         и предыдущих значениях параметров.
    :return: вектор-столбец новых коэффициентов модели.
    """
    # if model_value is None:
    #     a_n = last_a + gamma * (v_obj - np.dot(grad.T, last_a))
    # else:
    a_n = last_a + gamma * (v_obj - model_value)
    return a_n


def algorithm_polyak(identifier, obj_value, gamma, w, i):
    # TODO: не работает, разобраться
    m = identifier.model
    last_a = np.array(m.value_a)[None, :].T
    last_tmp_a = identifier.tmp_a

    # grad = np.array([func(*m.value_x, *m.value_u, *last_tmp_a) for func in m.grad])[None, :].T
    # n = m.func_model(*m.value_x, *m.value_u, *last_tmp_a)

    grad = np.array([func(*m.value_x, *m.value_u, *m.value_a) for func in m.grad])[None, :].T
    n = m.func_model(*m.value_x, *m.value_u, *m.value_a)

    last_tmp_a = np.array(last_tmp_a)[None, :].T
    tmp_an = last_tmp_a + gamma * (1 / (i ** 0.5)) * (1 / w) * grad * (obj_value - n)
    identifier.tmp_a = tmp_an[:, 0]
    a_n = last_a + (1 / i) * (tmp_an - last_a)
    return a_n[:, 0]


# TODO: унивесальная функция для 6.6.3, 6.6.6 и 6.6.8 и 6.6.10
def lsm_linear_parametrization(identifier, obj_value, w, lbd=1.):
    # TODO: не работает для нелинейных моделей, хз почему. 6.6.13!!!!
    """
    Квадратичный критерий (критерий наименьших квадратов).
    
    Если w == sigma**2 (дисперсия случайной ошибки):
        Классический алгоритм наименьших квадратов (МНК) (6.6.3, 6.6.6).
        
        Функция подходит для использования с некоррелированной неравноточной выборкой 
        (сигма различна для каждого измерения) и с некоррелированной равноточной выборкой 
        (сигма одинакова).
        
        В общем случае используется следующие выражения для расчетов:
        a(n) = a(n-1) + g_n * (eta(n) - fi(u(n)).T * a(n-1))
        g_n = (K_a(n-1) * fi(u(n))) / (sigma(n)**(-2) + fi(u(n)).T * K_a(n-1) * fi(u(n)))
        K_a(n) = (E - g_n * fi(u(n)).T) * K_a(n-1)
        n = n_0 + 1, n_0 + 2, ...
        где: a(n) - вектор столбец новых параметров модели;
             a(n-1) - вектор столбец параметров на предыдущей итерации;
             g_n - коэффициент gamma;
             eta(n) - измерение выхода объекта;
             fi(u(n)) - вектор столбец базисных фунций (параметр a входит линейно);
             K_a(n-1) - корреляционная матрица;
             E - единичная матрица;
             n - номер итерации (измерения).
        
    Если w - произвольный вес ( == p):
        Квадратичный критерий с произвольным весом (6.6.8).
    
    Если lbd == 1:
        Забывания информациии нет.
        
    Если lbd != 1:
        Алгоритм с забыванием информации.
        lbd рекомендуется выбирать из интервала 0.9 <= lbd <= 0.995.
        В этом вчлучае в качестве веса в квадратичном критерии будет выступать:
            p**(-1) * lbd**(n-i)
        а в критерии наименьших квадратов:
            sigma**(-2) * lbd**(n-i)
        В рекурентном алгоритме при расчете g_n вместо sigma(n)**(-2) будет стоять:
            w * lbd
        А матрица Г_a(n) будет домножяться на lbd**(-1).
    
    Адаптивный алгоритм подстройки параметров модели.
    Используется при линейной параметризации.
    
    Перед началом коррекции параметров необходимо накопить n_0 измерений. 
    Число n_0 должно быть больше числа параметров, т.е. если в модели три параметра
    a0, a1 и a2, минимальное число измерений должно быть равно 3.
    
    :param w: вес.
    :param lbd: коэффициент забывания информации.
    :param identifier: экземпляр класса Identifier.
    :param obj_value: измерение выхода объекта, float.
    :return: список новых параметров модели.
    """
    print('-'*20)
    if identifier.matrix is not None:
        k = identifier.matrix
    else:
        k = find_matrix(identifier.grad, identifier.sigma)
    m = identifier.model
    print(k)
    # a_n, k_n = adaptive_lsm(m, k, obj_value, w, lbd=lbd)

    grad = np.array([func(*m.value_x, *m.value_u, *m.value_a) for func in m.grad])[None, :].T
    print(grad)
    gamma = (k @ grad) / (w * lbd + grad.T @ k @ grad)
    # gamma = get_gamma(k, grad, w * lbd)
    print(gamma)
    last_a = np.array(m.value_a)[None, :].T
    print(last_a)
    model_value = m.func_model(*m.value_x, *m.value_u, *m.value_a)
    a_n = last_a + gamma * (obj_value - model_value)
    # a_n = get_a(last_a, gamma, obj_value, n)
    print(model_value)
    print(a_n)
    # k_n = (np.eye(len(gamma)) - gamma @ grad.T) @ k * (1 / lbd)
    k_n = (np.eye(len(gamma)) - gamma @ grad.T) * k * (1 / lbd)

    identifier.matrix = k_n

    return a_n[:, 0]


def get_covariance_matrix(identifier):
    """
    Функция расчета ковариационной матрицы при использовании 
    классического метода наименьших квадратов.
    
    Матрица имеет размерность m * m, где m - количество коэффициентов a.
    
    Рассчитыванется по формуле:
        K_a(n0) = sum{1..n0}(sigma**(-2) * fi(u(n)) * fi(u(n)).T)
        где: n0 - количество накопленных измерений, n0 >= m.
    
    :param identifier: экземпляр класса Identifier.
    :return: ковариационная матрица.
    """
    n_0 = identifier.n_0
    grad = np.array(identifier.grad)
    sigma = identifier.sigma
    if n_0 != len(grad):
        raise ValueError("n0 не совпадает с количеством векторов-градиентов (grad)")
    if (type(sigma) is list) and (len(sigma) != n_0):
        raise ValueError("n0 не совпадает с количеством среднеквадратичных ошибок измерений (sigma)")

    if type(sigma) in [float, int]:
        # k = np.sum([sigma ** (-2) * (grad[None, i].T * grad[None, i]) for i in range(n_0)], 0)
        k = np.sum([sigma ** (-2) * (grad[None, i].T @ grad[None, i]) for i in range(n_0)], 0)
    elif type(sigma) in [list, np.array]:
        # k = np.sum([sigma[i] ** (-2) * (grad[None, i].T * grad[None, i]) for i in range(n_0)], 0)
        k = np.sum([sigma[i] ** (-2) * (grad[None, i].T @ grad[None, i]) for i in range(n_0)], 0)
    else:
        raise ValueError("Неверный тип sigma")

    return k  # np.linalg.inv(k)


def find_matrix(grad, w):
    # TODO: переделать документацию
    """
    Функция расчета ковариационной матрицы при использовании 
    классического метода наименьших квадратов.

    Матрица имеет размерность m * m, где m - количество коэффициентов a.

    Рассчитыванется по формуле:
        K_a(n0) = sum{1..n0}(sigma**(-2) * fi(u(n)) * fi(u(n)).T)
        где: n0 - количество накопленных измерений, n0 >= m.

    :return: ковариационная матрица.
    """
    n_0 = len(grad)
    grad = np.array(grad)
    # sigma = identifier.sigma
    # if n_0 != len(grad):
    #     raise ValueError("n0 не совпадает с количеством векторов-градиентов (grad)")
    if (type(w) is list) and (len(w) != n_0):
        raise ValueError("n0 не совпадает с количеством среднеквадратичных ошибок измерений (sigma)")

    if type(w) in [float, int]:
        # k = np.sum([sigma ** (-2) * (grad[None, i].T * grad[None, i]) for i in range(n_0)], 0)
        k = np.sum([(1 / w) * (grad[i][None, :].T @ grad[i][None, :]) for i in range(n_0)], 0)
    elif type(w) in [list, np.array]:
        # k = np.sum([sigma[i] ** (-2) * (grad[None, i].T * grad[None, i]) for i in range(n_0)], 0)
        k = np.sum([(1 / w[i]) * (grad[i][None, :].T @ grad[i][None, :]) for i in range(n_0)], 0)
    else:
        raise ValueError("Неверный тип sigma")

    return k  # np.linalg.inv(k)


def lsm_linear_parametrization_efi(identifier, obj_value, p, lbd=1.):
    """
    Квадратичный критерий с замененными коэффициентами (+забывание информации) 
    (6.6.8, 6.6.10б 6.6.13 - нелинейный случай).
    Если lbd == 1:
        Квадратичный критерий с произвольными весами.
        Тот же вид, что и в lsm_linear_parametrization, только заменены sigma(n)**(-2)
        на p**(-1) и корреляционная матрица K_a(n) на матрицу Г_a(n).
        
    Если lbd != 1:
        Рекурсивный алгоритм с забыванием информации.
        lbd рекомендуется выбирать из интервала 0.9 <= lbd <= 0.995.
        В этом вчлучае в качестве веса в квадратичном критерии будет выступать:
            p**(-1) * lbd**(n-i)
        В рекурентном алгоритме при расчете g_n вместо sigma(n)**(-2) будет стоять:
            p * lbd
        А матрица Г_a(n) будет домножяться на lbd**(-1).
    
    Здесь необходимо также накапливать измерения (количество >= числу параметров a).
    
    После накопления измерений необходимо рассчитать начальную матрицу Г_a(0):
        Г_a(0) = [sum{1..n0}(p_i**(-1)*fi(u(n))*fi(u(n)).T)]**(-1)
        здесь:
            n0          - количество накопленных измерений;
            [...]**(-1) - взятие обратной матрицы;
            p_i**(-1)   - произвольный коэффициент;
            fi(u(n))    - вектор базисных функций (для линейного случая) или 
                          вектор производных по параметру (для нелинейного случая)
    
    :param identifier: экземпляр класса Identifier.
    :param obj_value: измерение выхода объекта.
    :param p: произвольный вес.
    :param lbd: коэффициент забывания информации, если == 1 - забывания нет.
    :return: список новых коэффициентов модели.
    """
    if identifier.matrix is not None:
        g = identifier.matrix
    else:
        g = find_matrix(identifier.grad, identifier.sigma)
    m = identifier.model

    a_n, g_n = adaptive_lsm(m, g, obj_value, p, lbd=lbd)

    # grad = np.array([func(*m.value_x, *m.value_u, *m.value_a) for func in m.grad])[None, :].T
    # gamma = get_gamma(g, grad, p*lbd)
    #
    # last_a = np.array(m.value_a)[None, :].T
    #
    # n = m.func_model(*m.value_x, *m.value_u, *m.value_a)
    # a_n = get_a(last_a, gamma, obj_value, n)
    #
    # g_n = (np.eye(len(gamma)) - gamma @ grad.T) @ g * (1 / lbd)

    identifier.matrix = g_n

    return a_n[:, 0]


def adaptive_lsm(model, matrix, obj_value, w, lbd=1.):
    # TODO: документация
    grad = np.array([func(*model.value_x, *model.value_u, *model.value_a) for func in model.grad])[None, :].T
    gamma = get_gamma(matrix, grad, w * lbd)

    last_a = np.array(model.value_a)[None, :].T

    n = model.func_model(*model.value_x, *model.value_u, *model.value_a)
    a_n = get_a(last_a, gamma, obj_value, n)

    k_n = (np.eye(len(gamma)) - gamma @ grad.T) @ matrix * (1 / lbd)
    return a_n, k_n


def get_g_matrix(identifier):
    # TODO: документация
    n_0 = identifier.n_0
    grad = np.array(identifier.grad)
    p = identifier.sigma
    if n_0 != len(grad):
        raise ValueError("n0 не совпадает с количеством векторов-градиентов (grad)")
    if (type(p) is list) and (len(p) != n_0):
        raise ValueError("n0 не совпадает с количеством среднеквадратичных ошибок измерений (sigma)")

    if type(p) in [float, int]:
        g = np.sum([(1 / p) * (grad[None, i].T @ grad[None, i]) for i in range(n_0)], 0)
    elif type(p) in [list, np.array]:
        g = np.sum([(1 / p[i]) * (grad[None, i].T @ grad[None, i]) for i in range(n_0)], 0)
    else:
        raise ValueError("Неверный тип sigma")

    print(g)
    return g


def g1(x, x1):
    return 4 + 1 * x - 5 * np.sin(x1)  # + 5 * np.cos(x) + np.random.uniform(-1, 1)  # + np.random.uniform(-1, 1)


def main():
    model = create_model("a_0+a_1*x(t-1)+a_2*sin(x(t-2))")  # +a_2*sin(x(t-1))
    model1 = create_model("a_0+a_1*x(t-1)+a_2*sin(x(t-2))")  # +a_2*sin(x(t-1))
    # model2 = create_model("a_0+a_1*x(t-1)+a_2*sin(x(t-2))")  # +a_2*sin(x(t-1))
    identifier = Identifier()
    identifier1 = Identifier()
    model.initialization(a0=0, a1=0, a2=0)
    identifier.model = model
    identifier.n_0 = 5
    identifier.sigma = 0.3

    model1.initialization(a0=0, a1=0, a2=0)
    identifier1.model = model1
    identifier1.n_0 = 5
    identifier1.sigma = 0.3

    # model2.initialization(a0=1, a1=1, a2=1)
    # identifier2 = Identifier()
    # identifier2.model = model2
    # identifier2.n_0 = 5
    # identifier2.sigma = 0.3
    # identifier2.tmp_a = [1,1,1]

    obj = [0.0]
    a0 = [0]
    a1 = [0]
    a2 = [0]

    a01 = [0]
    a11 = [0]
    a21 = [0]

    # a02 = [1]
    # a12 = [1]
    # a22 = [1]

    t_a = [0]
    last_a = []
    grad = []
    m = [0.]
    m1 = [0.]
    # m2 = [0.]
    model.value_x = [0.]
    model1.value_x = [0.]

    t = 0
    for i in range(30):
        if i == 0:
            obj_value = g1(obj[-1], 0)
        else:
            obj_value = g1(obj[-1], obj[-2])
        # model.value_x = [obj_value]
        # model.value_u = [t]
        if i < identifier.n_0:
            grad.append([func(*model.value_x, *model.value_u, *model.value_a) for func in model.grad])
            model.value_a = [a0[-1], a1[-1], a2[-1]]  # , a2[-1]
            model1.value_a = [a01[-1], a11[-1], a21[-1]]  # , a2[-1]
            a0.append(a0[-1])
            a1.append(a1[-1])
            a2.append(a2[-1])
            a01.append(a01[-1])
            a11.append(a11[-1])
            a21.append(a21[-1])
            # m.append(0.)
            print(grad[-1])
        else:
            if i == identifier.n_0:
                print('-'*20)
                identifier.grad = grad
                identifier1.grad = grad
                # k = get_covariance_matrix(identifier)
                # identifier.matrix = k
            a_n = lsm_linear_parametrization(identifier, obj_value, 0.3)
            model.value_a = list(a_n)
            a_n1 = lsm_linear_parametrization_efi(identifier1, obj_value, 0.3, lbd=0.99)
            model1.value_a = list(a_n1)
            # m.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
            a0.append(a_n[0])
            a1.append(a_n[1])
            a2.append(a_n[2])
            a01.append(a_n1[0])
            a11.append(a_n1[1])
            a21.append(a_n1[2])
        # a_n2 = algorithm_polyak(identifier2, obj_value, 0.01, 0.1, i + 1)
        # model2.value_a = list(a_n2)
        # a02.append(a_n2[0])
        # a12.append(a_n2[1])
        # a22.append(a_n2[2])
        t += 0.1
        m.append(model.func_model(*model.value_x, *model.value_u, *model.value_a))
        m1.append(model1.func_model(*model1.value_x, *model1.value_u, *model1.value_a))
        # m2.append(model2.func_model(*model2.value_x, *model2.value_u, *model2.value_a))
        if i == 0:
            model.value_x = [obj_value, .0]
            model1.value_x = [obj_value, .0]
            # model2.value_x = [obj_value, .0]
        else:
            model.value_x = [obj_value, obj[-1]]
            model1.value_x = [obj_value, obj[-1]]
            # model2.value_x = [obj_value, obj[-1]]
        obj.append(obj_value)
        t_a.append(t)

    # print(t_a)
    print(obj)
    print(m)
    print(a0)
    print(a1)
    print(a2)
    print('-'*20)
    print(m1)
    print(a01)
    print(a11)
    print(a21)
    # print('-' * 20)
    # print(a02)
    # print(a12)
    # print(a22)

    plt.plot(t_a, a0, label='a0')
    plt.plot(t_a, a1, label='a1')
    plt.plot(t_a, a2, label='a2')
    plt.plot(t_a, obj, label='Объект')
    plt.plot(t_a, m, label='Модель')
    plt.plot(t_a, a01, label='a01')
    plt.plot(t_a, a11, label='a11')
    plt.plot(t_a, a21, label='a21')
    plt.plot(t_a, m1, label='Модель1')

    # plt.plot(t_a, a02, label='a02')
    # plt.plot(t_a, a12, label='a12')
    # plt.plot(t_a, a22, label='a22')
    # plt.plot(t_a, m2, label='Модель2')

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
