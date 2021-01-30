import itertools

with open('../Lexica/most_common_1000.txt') as vocab_io:
    vocab = vocab_io.readlines()
    with open('../Lexica/most_common_1000_two_grams.txt', 'w') as two_grams_io:
        two_grams = itertools.permutations(vocab, 2)
        two_grams_io.write('\n'.join([
            ' '.join([word.strip() for word in pair])
            for pair in two_grams
        ]))