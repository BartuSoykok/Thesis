from abc import ABC, abstractmethod

class ModelRunner(ABC):

    def __init__(self, model, debug: bool = False):
        self.model = model
        self.debug = debug

    @abstractmethod
    def reset_outputs(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass
    