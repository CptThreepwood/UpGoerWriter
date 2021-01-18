from scipy.spatial import distance

def similarity(w1, w2):
    return distance.cosine(w1, w2)


def top_matches(word, lexicon, n=4):
    return sorted([
        (label, similarity(word, vector))
        for label, vector in lexicon.items()
    ], key=lambda x: x[1])[:4]


def closest_match(word, lexicon):
    lexicon_vectors = lexicon.items()
    closest = lexicon_vectors[0][0]
    distance = similarity(lexicon_vectors[0][1], word)
    for (label, vector) in lexicon_vectors:
        sim = similarity(word, vector)
        if sim < distance:
            closest = label
            distance = sim
    return closest


