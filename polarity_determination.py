from hazm import *


def load_lexicons():
    dataheart_lexicon_pos = open('./resources/dataheart_lexicon/positive_words.txt').read().split('\n')
    dataheart_lexicon_neg = open('./resources/dataheart_lexicon/negative_words.txt').read().split('\n')
    textmining_lexicon_pos = open('./resources/text_mining_lexicon/positive.txt').read().split('\n')
    textmining_lexicon_neg = open('./resources/text_mining_lexicon/negative.txt').read().split('\n')
    infogain_lexicon_pos = open('./resources/infogain_features/positive.txt').read().split('\n')
    infogain_lexicon_neg = open('./resources/infogain_features/negative.txt').read().split('\n')

    pos_words = dataheart_lexicon_pos + textmining_lexicon_pos + infogain_lexicon_pos
    neg_words = dataheart_lexicon_neg + textmining_lexicon_neg + infogain_lexicon_neg
    return pos_words, neg_words


def text_polarity(text):
    positive_words, negative_words = load_lexicons()
    words = word_tokenize(text)
    score = 0

    for word in words:
        if word in positive_words:
            score += 1
        if word in negative_words:
            score += -1

    if score >= 1:
        label = 1.0
    elif score == 0:
        label = 0.0
    else:
        label = -1.0
    return label

    # reading lexicons using spark
    # dataheart_lexicon_pos = spark.read.text('./resources/dataheart_lexicon/positive_words.txt')
    # dataheart_lexicon_neg = spark.read.text('./resources/dataheart_lexicon/negative_words.txt')
    # textmining_lexicon_pos = spark.read.text('./resources/text_mining_lexicon/positive.txt')
    # textmining_lexicon_neg = spark.read.text('./resources/text_mining_lexicon/negative.txt')
    # infogain_lexicon_pos = spark.read.text('./resources/infogain_features/positive.txt')
    # infogain_lexicon_neg = spark.read.text('./resources/infogain_features/negative.txt')
    #
    # pos_words = reduce(DataFrame.unionAll, [dataheart_lexicon_pos, textmining_lexicon_pos, infogain_lexicon_pos])
    # neg_words = reduce(DataFrame.unionAll, [dataheart_lexicon_neg, textmining_lexicon_neg, infogain_lexicon_neg])
    # neg_words.printSchema()
