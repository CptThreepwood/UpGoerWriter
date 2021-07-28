from abc import ABC, abstractmethod


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, sentence: str):
        pass

    @abstractmethod
    def __init__(self, model_name: str, lexicon: list[str]):
        pass
