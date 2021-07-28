from pathlib import Path
from typing import Optional

import spacy

LEXICON_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_LEXICON = "most_common_1000.txt"


SUPPLEMENTARY_VECTORS = {
    "aren't": ["are", "not"],
    "couldn't": ["could", "not"],
    "hadn't": ["had", "not"],
    "haven't": ["have", "not"],
    "he'd": ["he", "would"],
    "he's": ["he", "is"],
    "i'd": ["I", "would"],
    "i'll": ["I", "will"],
    "i'm": ["I", "am"],
    "isn't": ["is", "not"],
    "i've": ["I", "have"],
    "she'd": ["she", "would"],
    "she's": ["she", "is"],
    "shouldn't": ["should", "not"],
    "there's": ["there", "is"],
    "they'd": ["they", "would"],
    "they're": ["they", "are"],
    "wasn't": ["was", "not"],
    "we'll": ["we", "will"],
    "we're": ["we", "are"],
    "we've": ["we", "have"],
    "weren't": ["were", "not"],
    "what's": ["what", "is"],
    "won't": ["will", "not"],
    "wouldn't": ["would", "not"],
    "you'd": ["you", "would"],
    "you've": ["you", "have"],
}


def get_recipe(word: str):
    if word not in SUPPLEMENTARY_VECTORS:
        raise KeyError("No recipe for unknown word: {0}".format(word))

    return SUPPLEMENTARY_VECTORS[word]


def load(target_lexicon: str):
    with open(Path(LEXICON_DIR) / (target_lexicon + ".txt")) as lexicon_io:
        return [w.strip() for w in lexicon_io.readlines()]


class SentenceChecker:
    model: spacy.language.Language
    matcher: spacy.matcher.Matcher
    lemma: list[str]

    def __init__(
        self, spacy_model: str = DEFAULT_MODEL, vocab_file: str = DEFAULT_LEXICON
    ):
        self.model = spacy.load(spacy_model)
        with open((LEXICON_DIR / vocab_file).with_suffix(".txt"), "r") as lex_io:
            self.lemmas = [
                token.lemma_
                for line in lex_io.readlines()
                for token in self.model(line)
            ]
        self.matcher = spacy.matcher.Matcher(self.model.vocab)
        self.matcher.add("complex_match", [[{"LEMMA": {"NOT_IN": self.lemmas}}]])

    def complex_spans(
        self, sentence: str, char_span: Optional[tuple[int, int]] = None
    ) -> list[spacy.tokens.Span]:
        doc = self.model(sentence)
        if char_span:
            span = doc.char_span(*char_span)
        else:
            span = doc
        return self.matcher(span, as_spans=True)

    def is_sentence_simple(
        self, sentence: str, char_span: Optional[tuple[int, int]] = None
    ) -> bool:
        return len(self.complex_spans(sentence, char_span)) == 0
