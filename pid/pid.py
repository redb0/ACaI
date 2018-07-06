from typing import Union, Tuple


def in_interval():  # ???
    pass


class PID:
    def __init__(self, p=0., i=0., d=0., f=1.):
        self._msr_frequency = float(f)  # measuring_frequency

        self._k_p = p
        self._k_i = i / self._msr_frequency
        self._k_d = d * self._msr_frequency

        self._PoE = True  # True - PoE (Proportional on error), False - PoM (Proportional on measurement),
        # self.PoM = False
        self._DoE = True  # True - DoE (Derivative on error), False - DoM (Derivative on measurement). Derivative kick.

        self._u1 = 0.
        self._u2 = 0.

    def update(self, set_point, obj_value):
        pass

    def set_constraints(self, u1=0., u2=0.) -> None:
        if u1 < u2:
            self._u1 = u1
            self._u2 = u2
        else:
            self._u1 = 0.
            self._u2 = 0.

    def limit(self, u):
        # TODO: посмотреть какая операция быстрее ">" или ">="
        if u < self._u1:
            u = self._u1
        if u > self._u2:
            u = self._u2

        return u

    def set_coefficients(self, p=0., i=0., d=0.) -> None:
        self._k_p = p
        self._k_i = i / self._msr_frequency
        self._k_d = d * self._msr_frequency

    # def set_sampling_time(self, t=1):  # ????
    #     pass

    def set_measuring_frequency(self, f=1.) -> None:
        self._k_i = (self._k_i * self._msr_frequency)
        self._k_d = (self._k_d / self._msr_frequency)
        if f > 0:
            self._msr_frequency = float(f)
        else:
            self._msr_frequency = 1.
        self._k_i /= self._msr_frequency
        self._k_d *= self._msr_frequency

    # Свойства
    @property
    def k_p(self) -> Union[int, float]:
        return self._k_p

    @k_p.setter
    def k_p(self, value: Union[int, float]) -> None:
        self._k_p = value

    @property
    def k_i(self) -> Union[int, float]:
        return self._k_i

    @k_i.setter
    def k_i(self, value: Union[int, float]) -> None:
        self._k_i = value / self._msr_frequency

    @property
    def k_d(self) -> Union[int, float]:
        return self._k_d

    @k_d.setter
    def k_d(self, value: Union[int, float]) -> None:
        self._k_d = value * self._msr_frequency

    @property
    def sampling_time(self) -> float:
        return 1 / self._msr_frequency

    @property
    def measuring_frequency(self) -> float:
        return self._msr_frequency

    @property
    def constraints(self) -> Tuple[Union[int, float], Union[int, float]]:
        return self._u1, self._u2

    @property
    def derivative_on_error(self) -> bool:
        return self._DoE

    @derivative_on_error.setter
    def derivative_on_error(self, value: bool) -> None:
        self._DoE = value

    @property
    def proportional_on_error(self) -> bool:
        return self._PoE

    @proportional_on_error.setter
    def proportional_on_error(self, value: bool) -> None:
        self._PoE = value

