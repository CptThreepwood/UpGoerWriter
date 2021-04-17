import os
import glob
import tarfile

import boto3
import spacy

LOCAL_MODEL_DIR = os.environ['LOCAL_MOEDEL_DIR']
S3_BUCKET = os.environ['S3_BUCKET']

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


def download_model_from_s3(model_name: str, local_path: str) -> str:
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    local_compressed = f'{local_path}.tar.gz'

    # download model is the archive doesn't exist
    if not os.path.exists(local_compressed):
        print('Downloading model...')
        s3_prefix = f'models/{model_name}.tar.gz'
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, s3_prefix, local_compressed)

    with tarfile.open(local_compressed) as f:
        f.extractall(path=local_path)

    return local_path


def get_model_path(model_name):
    local_path = os.path.join(LOCAL_MODEL_DIR, model_name)
    if os.path.exists(local_path):
        return local_path
    return download_model_from_s3(model_name, local_path)


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
        self.language_model = spacy.load(get_model_path(spacy_model))
        with open(os.path.join(LEXICON_DIR, target_lexicon) + '.txt') as lexicon_io:
            self.target_lexicon = spacy.tokens.Doc(
                self.language_model.vocab,
                words=[w.strip() for w in lexicon_io.readlines()]
            )
        for word in self.target_lexicon:
            if not word.has_vector:
                ## This affects words in target_lexicon, presumably because everything's pointers
                self.language_model.vocab.set_vector(word.norm_, self.create_vector(word.norm_))
        self.lexicon_norms = [word.norm_ for word in self.target_lexicon]
        self.lexicon_lemmas = [word.lemma_ for word in self.target_lexicon]

    def closest_target(self, word):
        '''Find the closest matching word in the target lexicon'''
        # Keep punctuation or numbers
        if word.is_punct or word.is_digit or word.is_space:
            return word.orth_

        # Pass through if the input isn't in the spacy model
        if not word.has_vector:
            print('No vector for {0} ({1})'.format(word.orth_, word.norm_))
            return word.orth_

        # If the word is already in the target lexicon, return it
        if word.norm_ in self.lexicon_norms or word.lemma_ in self.lexicon_lemmas:
            return word.orth_

        # Find the word in the target with maximum spacy similarity (cosine)
        best_match = max([
            (target.text, word.similarity(target))
            for target in self.target_lexicon
        ], key=lambda x: x[1])[0]
        print('{0} -> {1}'.format(word, best_match))
        return best_match

    def translate(self, document):
        '''
            Translate a document into the target lexicon
            Current default translation mechanism is one word at a time
        '''
        doc = self.language_model(document)
        token_end = 0
        simplified = ''
        for token in doc:
            token_start = token.idx
            simplified += document[token_end:token_start]
            token_end = token_start + len(token.text)
            simplified += self.closest_target(token)

        return simplified

