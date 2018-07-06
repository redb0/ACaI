from pid.pid import PID


class RecurrentPID(PID):
    """
    Реализация рекурентной формулы PID-регулятора.
    v(t) = u(t-1) + P + I + D;
    v(t) - идеальное управление (управление без ограничений).
    Пропорциональная составляющая : P = Kp * [e(t) - e(t-1)];
    Интегральная составляющая     : I = Ki * e(t);
    Дифференциальная составляющая : D = Kd * [e(t) - 2 * e(t-1) + e(t-2)];
    Сигнал рассоглаования         : e(t) = x*(t) - x(t);
    Управление для подачи на объект определяется после накладывания ограничений:
    u(t) = u1  , при v(t) <= u1;
    u(t) = v(t), при u1 < v(t) < u2;
    u(t) = u2  , при v(t) >= u2;
    
    Поддерживается две модификации:
    1) Пропорциональная составляющая, основанная на измерениях выхода объекта. 
        Позволяет избежать перерегулирования. Активация по флагу p_on_e = False.
        v(t) = u(t-1) - P + I + D.
        P = Kp * [x(t) - x(t-1)].
    2) Дифференциальная составляющая, основанная на измерениях выхода объекта.
        Позволяет избежать "производного удара" (Derivative kick). 
        Активация по флагу d_on_e = False.
        v(t) = u(t-1) + P + I - D.
        D = Kd * [x(t) - 2 * x(t-1) + x(t-2)].
    Модификации можно совмещать.
    
    Примеры:
    1) Создание PID-регулятора (стандартная рекурентная формула) 
        с коэффициентами p = 1, i = 0.5, d = 0.1, частотой 
        измерений 5 (время дискретизации 0.2 секунды):
        >>> RecurrentPID(1., 0.5, 0.1, 5, True, True)
        RecurrentPID(1.0, 0.1, 0.5, 5.0, True, True)
        или:
        >>> RecurrentPID(1., 0.5, 0.1, 5)
        RecurrentPID(1.0, 0.1, 0.5, 5.0, True, True)
    2) Создание PID-регулятора (с пропорциональной составляющей, 
        основанной на измерениях выхода объекта):
        >>> RecurrentPID(1., 0.5, 0.1, 5, False, True)
        RecurrentPID(1.0, 0.1, 0.5, 5.0, False, True)
    3) Создание PID-регулятора (с дифференциальной составляющей, 
        основанной на измерениях выхода объекта):
        >>> RecurrentPID(1., 0.5, 0.1, 5, True, False)
        RecurrentPID(1.0, 0.1, 0.5, 5.0, True, False)
    4) Создание PID-регулятора (с совмещенными модификациями):
        >>> RecurrentPID(1., 0.5, 0.1, 5, False, False)
        RecurrentPID(1.0, 0.1, 0.5, 5.0, False, False)
    """
    def __init__(self, p, i, d, f=1., p_on_e=True, d_on_e=True):
        super().__init__(p, i, d, f)

        self.proportional_on_error = p_on_e
        self.derivative_on_error = d_on_e

        self.last_u = 0
        self.last_err = 0
        self.last_last_err = 0

        self.last_obj_value = 0
        self.last_last_obj_value = 0

    def initialization(self, obj_value, last_obj_value, u) -> None:
        """
        Инициализацция регулятора имеющимися значениями.
        Используется, когда регулятор внедряется в работающею систему 
        и уже имеются измерения входов и выходов объекта.
        :param obj_value      : значение выхода объекта, x(t).
        :param last_obj_value : значение выхода объекта, x(t-1).
        :param u              : значение подаваемого управления.
        :return: None
        """
        self.last_u = self.limit(u)
        self.last_obj_value = obj_value
        self.last_last_obj_value = last_obj_value

    def update(self, set_point, obj_value):
        # u = self.past_u + self.k_p * (err - self.past_err) + self.k_i * err + self.k_d * (err - 2*self.past_err + self.past_past_err)

        err = set_point - obj_value

        u = self.last_u + self.k_i * err

        if self._PoE:
            u += self.k_p * (err - self.last_err)
        else:
            u -= self.k_p * (obj_value - self.last_obj_value)

        if self._DoE:
            u += self.k_d * (err - 2 * self.last_err + self.last_last_err)
        else:
            u -= self.k_d * (obj_value - 2 * self.last_obj_value + self.last_last_obj_value)

        # u = self.k_p * err + self.k_i * self.integral + self.k_d * (err - self.last_error)

        u = self.limit(u)

        self.last_last_err = self.last_err
        self.last_err = err
        self.last_last_obj_value = self.last_obj_value
        self.last_obj_value = obj_value

        return u

    def __repr__(self):
        return "RecurrentPID(%r, %r, %r, %r, %r, %r)" % (self.k_p, self.k_i, self.k_d,
                                                         self.measuring_frequency,
                                                         self.proportional_on_error, self.derivative_on_error)
