import fire
from .SpacyTranslator import Translator


def convert(
    model="en_core_web_lg",
    lexicon="most_common_1000",
    input_string="This is a test sentence",
):
    translator = Translator(model, lexicon)
    print(translator.translate(input_string))


if __name__ == "__main__":
    fire.Fire(convert)
