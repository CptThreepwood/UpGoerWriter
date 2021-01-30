import os

import spacy

LEXICON_DIR = '../Lexica'

class Translator:
    def __init__(self, spacy_model, target_lexicon):
        self.language_model = spacy.load(spacy_model)
        with open(os.path.join(LEXICON_DIR, target_lexicon)) as lexicon_io:
            self.target_lexicon = [
                self.language_model(word)
                for word in lexicon_io.readlines()
            ]

    def word_translate(self, word):
        if word in [t.text for t in self.target_lexicon]:
            print('exact')
            return word
        return max([
            (target.text, self.language_model(word).similarity(target))
            for target in self.target_lexicon
        ], key=lambda x: x[1])[0]

def most_common_thousand():
    model = spacy.load('en_core_web_lg')
    with open('allowedWords.txt', 'r') as word_io:
        return word_io.readlines()


