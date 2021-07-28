import spacy

from translators.BaseTranslator import BaseTranslator

import lexicon
from .load_model import get_model_path


class Translator(BaseTranslator):
    """
    Translator Object contains a vector model, target lexicon and techniques for performing the translation
    """

    def __init__(self, model_name="en_core_web_lg", target_lexicon="most_common_1000"):
        """Create a translator using a spacy model (for vectors) and a target lexicon"""
        try:
            ## Try to load the model from a package
            self.language_model = spacy.load(model_name)
        except OSError as err:
            ## If there is no matching package, try to load from local cache
            self.language_model = spacy.load(get_model_path(model_name))

        self.target_lexicon = spacy.tokens.Doc(
            self.language_model.vocab, words=lexicon.load(target_lexicon)
        )
        for word in self.target_lexicon:
            if not word.has_vector:
                ## This affects words in target_lexicon, presumably because everything's pointers
                self.language_model.vocab.set_vector(
                    word.norm_, self.create_vector(word.norm_)
                )
        self.lexicon_norms = [word.norm_ for word in self.target_lexicon]
        self.lexicon_lemmas = [word.lemma_ for word in self.target_lexicon]

    def create_vector(self, unknown_word):
        """Create vectors from recipes for known missing words"""
        recipe = lexicon.get_recipe(unknown_word)
        for w in recipe:
            if not self.language_model.vocab[w].has_vector:
                raise KeyError("Recipe contains invalid component: {0}".format(w))
        return sum(self.language_model.vocab.get_vector(word) for word in recipe) / len(
            recipe
        )

    def closest_target(self, word):
        """Find the closest matching word in the target lexicon"""
        # Keep punctuation or numbers
        if word.is_punct or word.is_digit or word.is_space:
            return word.orth_

        # Pass through if the input isn't in the spacy model
        if not word.has_vector:
            print("No vector for {0} ({1})".format(word.orth_, word.norm_))
            return word.orth_

        # If the word is already in the target lexicon, return it
        if word.norm_ in self.lexicon_norms or word.lemma_ in self.lexicon_lemmas:
            return word.orth_

        # Find the word in the target with maximum spacy similarity (cosine)
        best_match = max(
            [(target.text, word.similarity(target)) for target in self.target_lexicon],
            key=lambda x: x[1],
        )[0]
        print("{0} -> {1}".format(word, best_match))
        return best_match

    def translate(self, document):
        """
        Translate a document into the target lexicon
        Current default translation mechanism is one word at a time
        """
        doc = self.language_model(document)
        token_end = 0
        simplified = ""
        for token in doc:
            token_start = token.idx
            simplified += document[token_end:token_start]
            token_end = token_start + len(token.text)
            simplified += self.closest_target(token)

        return simplified
