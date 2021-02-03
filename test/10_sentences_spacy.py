import os
import sys
abs_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(abs_path, '..'))

from SpacyTranslation import Translator

with open(os.path.join(abs_path, './fixtures/10_sentences.txt')) as fixture_io:
    test_sentences = fixture_io.readlines()

print('Loading model...')
translator = Translator('en_core_web_lg', 'most_common_1000')
print('Translating...')

for sentence in test_sentences:
    print(80 * '-')
    print(sentence)
    simplified = translator.translate(sentence)
    print(simplified)
print(80 * '-')
