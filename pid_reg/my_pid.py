class PID:
    def __init__(self, p=0., i=0., d=0., f=10):
        self.k_p = p
        self.k_i = i
        self.k_d = d
        self.measuring_frequency = f
        # self.time = t

        self.d_t = 1 / self.measuring_frequency

        self.past_err = 0
        self.past_past_err = 0
        self.past_u = 0
        # self.err = 0
        # self.past_u = 0
        self.integral = 0
        # self.windup_guard = 20

        self.u1 = 0
        self.u2 = 0

        self.last_x = 0

    def clear(self):
        self.k_p = 0.
        self.k_i = 0.
        self.k_d = 0.
        self.measuring_frequency = 10
        # self.time = 5

        self.d_t = 1 / (self.measuring_frequency)

        self.past_err = 0
        self.past_past_err = 0
        self.past_u = 0
        # self.err = 0
        self.integral = 0

        self.u1 = 0
        self.u2 = 0

    def set_constraints(self, u1=0, u2=0):
        """"""
        if u1 >= u2:
            self.u1 = 0
            self.u2 = 0
        else:
            self.u1 = u1
            self.u2 = u2

    def update(self, set_point, x):
        err = set_point - x
        d_e = err - self.past_err
        # d_e = self.last_x - x
        self.integral = self.integral + (self.past_err + err) / 2
        # self.integral += err * self.d_t

        c_i = self.integral * (1 / self.d_t)
        # c_i = self.integral * self.d_t

        # if c_i < -self.windup_guard:
        #     c_i = -self.windup_guard
        # elif (c_i > self.windup_guard):
        #     c_i = self.windup_guard
        # if c_i < self.u1:
        #     c_i = self.u1
        # elif c_i > self.u2:
        #     c_i = self.u2

        c_d = d_e / self.d_t
        # c_d = d_e

        # c_i = self.err / (1 / self.measuring_frequency)
        # c_d = (self.err - self.past_err) * (1 / self.measuring_frequency)

        # u = self.k_p * err + (1 / self.k_i) * c_i + self.k_d * c_d

        # рекурентный вид
        u = self.past_u + self.k_p * (err - self.past_err) + self.k_i * err + self.k_d * (err - 2*self.past_err + self.past_past_err)
        # self.past_u = u
        # нерекурентный вид
        # u = self.k_p * err + self.k_i * self.integral + self.k_d * d_e  # работает по формуле из википедии

        # u = self.k_p * err + self.k_i * self.integral - self.k_d * d_e

        if self.u1 != self.u2:  # if self.u1 == self.u2 == 0 ограничения на управление нет.
            if u < self.u1:
                u = self.u1
                # self.integral = 0
            elif u > self.u2:
                u = self.u2
                # self.integral = 0

        self.past_past_err = self.past_err
        self.past_err = err
        self.past_u = u

        self.last_x = x

        return u

    # def standard_update(self):
    #     pass

    # def recurrent_update(self):
    #     pass

    def __repr__(self):
        # FIXME: проверить что означает форматированный вывод %r и что для дробных и целых.
        return "PID(%r, %r, %r, %r)" % (self.k_p, self.k_i, self.k_d, self.measuring_frequency)
