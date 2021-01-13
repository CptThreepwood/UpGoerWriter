import fire
from ModelLoader import Translator 

def convert(model, lexicon, input):
    translator = Translator(model, lexicon)
    for word in input.split():
        print(translator.word_translate(word.lower()))

if __name__ == '__main__':
    fire.Fire(convert)