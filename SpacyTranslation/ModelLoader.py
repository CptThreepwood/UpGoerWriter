import os

import spacy

LEXICON_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Lexica')

SUPPLEMENTARY_VECTORS = {
    "aren't": ['are', 'not'],
    "couldn't": ['could', 'not'],
    "hadn't": ['had', 'not'],
    "haven't": ['have', 'not'],
    "he'd": ['he', 'would'],
    "he's": ['he', 'is'],
    "i'd": ['I', 'would'],
    "i'll": ['I', 'will'],
    "i'm": ['I', 'am'],
    "isn't": ['is', 'not'],
    "i've": ['I', 'have'],
    "she'd": ['she', 'would'],
    "she's": ['she', 'is'],
    "shouldn't": ['should', 'not'],
    "there's": ['there', 'is'],
    "they'd": ['they', 'would'],
    "they're": ['they', 'are'],
    "wasn't": ['was', 'not'],
    "we'll": ['we', 'will'],
    "we're": ['we', 'are'],
    "we've": ['we', 'have'],
    "weren't": ['were', 'not'],
    "what's": ['what', 'is'],
    "won't": ['will', 'not'],
    "wouldn't": ['would', 'not'],
    "you'd": ['you', 'would'],
    "you've": ['you', 'have'],
}

class Translator:
    '''
        Translator Object contains a vector model, target lexicon and techniques for performing the translation
    '''

    def create_vector(self, unknown_word):
        '''Create vectors from recipes for known missing words'''
        if unknown_word not in SUPPLEMENTARY_VECTORS:
            raise KeyError("No recipe for unknown word: {0}".format(unknown_word))
        recipe = SUPPLEMENTARY_VECTORS[unknown_word]
        for w in recipe:
            if not self.language_model.vocab[w].has_vector:
                raise KeyError("Recipe contains invalid component: {0}".format(w))
        return sum(self.language_model.vocab.get_vector(word) for word in recipe) / len(recipe)

    def __init__(self, spacy_model='en_core_web_lg', target_lexicon='most_common_1000'):
        '''Create a translator using a spacy model (for vectors) and a target lexicon'''
        self.language_model = spacy.load(spacy_model)
        with open(os.path.join(LEXICON_DIR, target_lexicon) + '.txt') as lexicon_io:
            lexicon = [
                self.language_model.vocab[word.strip().lower()]
                for word in lexicon_io.readlines()
            ]
        for word in lexicon:
            if not word.has_vector:
                self.language_model.vocab.set_vector(word.norm_, self.create_vector(word.norm_))
        self.target_lexicon = {word.norm_: word for word in lexicon}

    def closest_target(self, word):
        '''Find the closest matching word in the target lexicon'''
        if word.norm_ in self.target_lexicon:
            return word.orth_
        return max([
            (target.orth_, word.similarity(target))
            for target in self.target_lexicon.values()
        ], key=lambda x: x[1])[0]

    def translate(self, document):
        '''
            Translate a document into the target lexicon
            Current default translation mechanism is one word at a time
        '''
        doc = self.language_model(document)
        return ' '.join([self.closest_target(token) for token in doc])

