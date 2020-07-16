from abc import abstractmethod, ABC


class Neuron(ABC):
    def __init__(self):
        self.method = "el"

        super().__init__()

    def change_method(self, method):
        if method == "rk":
            self.method = "rk"
        else:
            self.method = "el"

    @abstractmethod
    def stimulation(self, tmax, I, dt):
        pass

    @abstractmethod
    def create_copy(self):
        pass
