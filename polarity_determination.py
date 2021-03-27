'''
this module is for predicting text polarity using sensory lexicons and rule based methods
'''

from hazm import *
import pandas as pd
from nltk import ngrams
pos_model = POSTagger(model='./resources/hazm_resources/postagger.model')


def load_lexicons():
    dataheart_lexicon_pos = open('./resources/dataheart_lexicon/positive_words.txt').read().split('\n')
    dataheart_lexicon_neg = open('./resources/dataheart_lexicon/negative_words.txt').read().split('\n')

    textmining_lexicon_pos = open('./resources/text_mining_lexicon/positive.txt').read().split('\n')
    textmining_lexicon_neg = open('./resources/text_mining_lexicon/negative.txt').read().split('\n')

    infogain_snapfood_pos = open('./resources/infogain_snapfood/positive.txt').read().split('\n')
    infogain_snapfood_neg = open('./resources/infogain_snapfood/negative.txt').read().split('\n')

    infogain_digikala_pos = open('./resources/infogain_digikala/positive_words.txt').read().split('\n')
    infogain_digikala_neg = open('./resources/infogain_digikala/negative_words.txt').read().split('\n')

    print("senti fars lexicon")
    sentifars = pd.read_excel('./resources/SentiFars_lexicon/SentiFars_lexicon.xlsx')
    print(sentifars)

    pos_words = dataheart_lexicon_pos + textmining_lexicon_pos + infogain_snapfood_pos + infogain_digikala_pos
    neg_words = dataheart_lexicon_neg + textmining_lexicon_neg + infogain_snapfood_neg + infogain_digikala_neg
    return pos_words, neg_words


def text_polarity(text, window=2):
    positive_words, negative_words = load_lexicons()
    words = word_tokenize(text)  # use Hazm tokenizer to get tokens
    bigrams = ngrams(words, 2)
    part_of_speech = pos_model.tag(words)
    score = 0

    for index, word in enumerate(words):
        if part_of_speech[index][1] == 'V':  # find negative verbs
            # print(word, part_of_speech[index], part_of_speech)
            if word[0] == 'ن':  # so the word is negative verb
                for i in range(window):
                    if index - i - 1 >= 0:
                        if words[index - i - 1] in positive_words:
                            score += -2
                        elif words[index - i - 1] in negative_words:
                            score += 2
        # check unigrams
        if word in positive_words:
            score += 1
        if word in negative_words:
            score += -1
        # check bigrams
        for grams in bigrams:
            token = ' '.join(grams)
            if token in positive_words:
                score += 1
            if token in negative_words:
                score += -1

    if score >= 1:
        label = 1.0
    elif score == 0:
        label = 0.0
    else:
        label = -1.0
    # print(part_of_speech, score, label)
    return label

    # # reading lexicons using spark
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
