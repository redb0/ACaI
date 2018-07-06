from typing import Union

from pid.pid import PID

# TODO: sampling time                                      +++
# TODO: derivative kick (d_on_e / d_on_m)                  +++
# TODO: измененеие параметров на ходу                      +++
# TODO: ограничение на управление и интеграл               +++ проверить
# TODO: on / off                                           ---
# TODO: инициализация                                      +++
# TODO: изменение направления (отрицательные коэффициенты) ????
# TODO: proportional on measurement (p_on_e / p_on_m)      +++


class StandardPID(PID):
    """
    Реализация стандартной формулы PID-регулятора.
    
    v(t) = P + I + D;
    v(t) - идеальное управление (управление без ограничений).
    Пропорциональная составляющая : P = Kp * e(t);
    Интегральная составляющая     : I = Ki * sum(e);
    Дифференциальная составляющая : D = Kd * [e(t) - e(t-1)];
    Сигнал рассоглаования         : e(t) = x*(t) - x(t);
    Управление для подачи на объект определяется после накладывания ограничений:
    u(t) = u1  , при v(t) <= u1;
    u(t) = v(t), при u1 < v(t) < u2;
    u(t) = u2  , при v(t) >= u2;
    
    Поддерживается две модификации:
    1) Пропорциональная составляющая, основанная на измерениях выхода объекта. 
        Позволяет избежать перерегулирования. Активация по флагу p_on_e = False.
        v(t) = - P + I + D.
        P = sum(Kp(n) * [x(n) - x(n-1)]).
    2) Дифференциальная составляющая, основанная на измерениях выхода объекта.
        Позволяет избежать "производного удара" (Derivative kick). 
        Активация по флагу d_on_e = False.
        v(t) = P + I - D.
        D = Kd * [x(t) - x(t-1)].
    Модификации можно совмещать.
    
    Примеры:
    1) Создание PID-регулятора (стандартная рекурентная формула) 
        с коэффициентами p = 1, i = 0.5, d = 0.1, частотой 
        измерений 5 (время дискретизации 0.2 секунды):
        >>> StandardPID(1., 0.5, 0.1, 5, True, True)
        StandardPID(1.0, 0.1, 0.5, 5.0, True, True)
        или:
        >>> StandardPID(1., 0.5, 0.1, 5)
        StandardPID(1.0, 0.1, 0.5, 5.0, True, True)
    2) Создание PID-регулятора (с пропорциональной составляющей, 
        основанной на измерениях выхода объекта):
        >>> StandardPID(1., 0.5, 0.1, 5, False, True)
        StandardPID(1.0, 0.1, 0.5, 5.0, False, True)
    3) Создание PID-регулятора (с дифференциальной составляющей, 
        основанной на измерениях выхода объекта):
        >>> StandardPID(1., 0.5, 0.1, 5, True, False)
        StandardPID(1.0, 0.1, 0.5, 5.0, True, False)
    4) Создание PID-регулятора (с совмещенными модификациями):
        >>> StandardPID(1., 0.5, 0.1, 5, False, False)
        StandardPID(1.0, 0.1, 0.5, 5.0, False, False)
    """
    def __init__(self, p, i, d, f=1., p_on_e: bool=True, d_on_e: bool=True):
        """
        Конструктор.
        :param p      : пропорциональный коэффициент.
        :param i      : интегральный коэффициент.
        :param d      : дифференциальный коэффициент.
        :param f      : частота измерений выхода объекта (кол-во в секунду).
        :param p_on_e : True - использование пропорциональной составляющей, 
                        основанной на сигнале рассогласования. 
                        False - пропорциональная составляющая, основанная 
                        на измерениях значения объекта.
        :param d_on_e : True - использование дифференциальной составляющей, 
                        основанной на сигнале рассогласования. 
                        False - дифференциальная составляющая, основанная 
                        на измерениях значения объекта.
        """
        super().__init__(p, i, d, f)
        self._PoE = p_on_e
        self._DoE = d_on_e

        self.last_error = 0
        self.last_obj_value = 0
        self.integral = 0

    def initialization(self, obj_value, u) -> None:
        """
        Инициализацция регулятора имеющимися значениями.
        Используется, когда регулятор внедряется в работающею систему 
        и уже имеются измерения входов и выходов объекта.
        :param obj_value : значение выхода объекта.
        :param u         : значение последнего поданного на объект управления.
        :return: None.
        """
        self.integral = self.limit(u)
        self.last_obj_value = obj_value

    def update(self, set_point: Union[int, float], obj_value: Union[int, float]) -> Union[int, float]:
        """
        Метод расчета управления.
        Используется стандартная дискретная формула PID-регулятора (см. описание класса).
        :param set_point: значение уставки, x*(t).
        :param obj_value: значение выхода объекта, x(t).
        :return: значение управления.
        """
        # u = self.k_p * err + self.k_i * self.integral - self.k_d * d_e

        err = set_point - obj_value
        self.integral += self.k_i * err
        self.integral = self.limit(self.integral)
        u = self.integral

        if self._PoE:
            u += self.k_p * err
        else:
            u -= self.k_p * (obj_value - self.last_obj_value)

        if self._DoE:
            u += self.k_d * (err - self.last_error)
        else:
            u -= self.k_d * (obj_value - self.last_obj_value)

        u = self.limit(u)

        # u = self.k_p * err + self.k_i * self.integral + self.k_d * (err - self.last_error)

        self.last_error = err
        self.last_obj_value = obj_value

        return u

    def __repr__(self):
        return "StandardPID(%r, %r, %r, %r, %r, %r)" % (self.k_p, self.k_i, self.k_d,
                                                        self.measuring_frequency,
                                                        self.proportional_on_error, self.derivative_on_error)

