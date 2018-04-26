class Identifier:
    def __init__(self, model=None, n=None, grad=None, sigma=None):
        self._n0 = n
        self._grad = grad
        self._sigma = sigma
        self._model = model

        self._matrix = None  # матрица

        self.tmp_a = None

    @property
    def n_0(self):
        return self._n0

    @n_0.setter
    def n_0(self, value: int):
        if value > 0:
            self._n0 = value
        else:
            raise AttributeError("Значение n0 должно быть больше 0")

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        # if value > 0:
        #     self._sigma = value
        # else:
        #     raise AttributeError("Значение sigma должно быть больше 0")

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value
