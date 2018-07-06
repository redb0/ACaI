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
    def __init__(self, p, i, d, f=1., p_on_e=True, d_on_e=True):
        super().__init__(p, i, d, f)
        self._PoE = p_on_e
        self._DoE = d_on_e

        self.last_error = 0
        self.last_obj_value = 0
        self.integral = 0

    def initialization(self, obj_value, u):
        #
        self.integral = self.limit(u)
        self.last_obj_value = obj_value

    def update(self, set_point, obj_value):
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

