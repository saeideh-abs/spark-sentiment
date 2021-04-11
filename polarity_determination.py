"""
this module is for predicting text polarity using sensory lexicons and rule based methods
"""

from hazm import *
import ast
from nltk import ngrams


def load_lexicons():
    root = '/home/mohammad/saeideh/spark-test/'
    dataheart_lexicon_pos = open(root + 'resources/dataheart_lexicon/positive_words.txt').read().split('\n')
    dataheart_lexicon_neg = open(root + 'resources/dataheart_lexicon/negative_words.txt').read().split('\n')

    textmining_lexicon_pos = open(root + 'resources/text_mining_lexicon/positive.txt').read().split('\n')
    textmining_lexicon_neg = open(root + 'resources/text_mining_lexicon/negative.txt').read().split('\n')

    infogain_snapfood_pos = open(root + 'resources/infogain_snapfood/positive.txt').read().split('\n')
    infogain_snapfood_neg = open(root + 'resources/infogain_snapfood/negative.txt').read().split('\n')

    infogain_digikala_pos = open(root + 'resources/infogain_digikala/positive_words.txt').read().split('\n')
    infogain_digikala_neg = open(root + 'resources/infogain_digikala/negative_words.txt').read().split('\n')

    sentifars_lexicon_pos = open(root + 'resources/SentiFars_lexicon/positive.txt').read().split('\n')
    sentifars_lexicon_neg = open(root + 'resources/SentiFars_lexicon/negative.txt').read().split('\n')

    pos_words = dataheart_lexicon_pos + textmining_lexicon_pos\
                + infogain_snapfood_pos + infogain_digikala_pos + sentifars_lexicon_pos
    neg_words = dataheart_lexicon_neg + textmining_lexicon_neg\
                + infogain_snapfood_neg + infogain_digikala_neg + sentifars_lexicon_neg
    return pos_words, neg_words


positive_words, negative_words = load_lexicons()
pos_model = POSTagger(model='/home/mohammad/saeideh/spark-test/resources/hazm_resources/postagger.model')


def find_label(text, advantages, disadvantages, window=2):
    pos_score, neg_score = text_polarity(text, window)
    advantages = ast.literal_eval(advantages)
    disadvantages = ast.literal_eval(disadvantages)
    advan_score, disadvan_score = calc_advan_disadvan_score(advantages, disadvantages)

    final_score = pos_score + neg_score + advan_score + disadvan_score

    if final_score >= 1:  # positive
        label = 1.0
    elif final_score == 0:  # neutral
        label = 2.0
    else:
        label = 0.0  # negative
    return label


def extract_features(text, advantages, disadvantages, window=2):
    pos_score, neg_score = text_polarity(text, window)
    advantages = ast.literal_eval(advantages)
    disadvantages = ast.literal_eval(disadvantages)
    advan_score, disadvan_score = calc_advan_disadvan_score(advantages, disadvantages)
    features = [pos_score, neg_score*(-1), advan_score, disadvan_score*(-1)]
    return features


def text_polarity(text, window=2):
    words = word_tokenize(text)  # use Hazm tokenizer to get tokens
    bigrams = ngrams(words, 2)
    trigrams = ngrams(words, 3)
    part_of_speech = pos_model.tag(words)
    pos_score = 0
    neg_score = 0

    # check unigrams
    for index, word in enumerate(words):
        if part_of_speech[index][1] == 'V':  # find negative verbs
            # print(word, part_of_speech[index], part_of_speech)
            if word[0] == 'ن':  # so the word is negative verb
                for i in range(window):
                    if index - i - 1 >= 0:
                        if words[index - i - 1] in positive_words:
                            neg_score += -2
                        elif words[index - i - 1] in negative_words:
                            pos_score += 2
        if word in positive_words:
            pos_score += 1
        if word in negative_words:
            neg_score += -1
    # check bigrams
    for grams in bigrams:
        bigram = ' '.join(grams)
        if bigram in positive_words:
            pos_score += 1
        if bigram in negative_words:
            neg_score += -1
    # check trigrams
    for grams in trigrams:
        trigram = ' '.join(grams)
        if trigram in positive_words:
            pos_score += 1
        if trigram in negative_words:
            neg_score += -1
    return pos_score, neg_score


def calc_advan_disadvan_score(advantages, disadvantages):
    exceptions = ['ندیدم', 'نبود', 'نیست', 'هیچی', 'هیچ', 'ندارد', 'ندارد ', 'نداره ', 'نداره', 'نداشت']
    advan_score = 0
    disadvan_score = 0

    for item in advantages:
        if item in exceptions:
            disadvan_score += -1
        else:
            advan_score += 1
    for item in disadvantages:
        if item in exceptions:
            advan_score += 1
        else:
            disadvan_score += -1

    return advan_score, disadvan_score


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

# # add lexicons using spark context addFile
# sc = SparkContext.getOrCreate()
#
#
# # def load_lexicons():
# sc.addFile("./resources/dataheart_lexicon/positive_words.txt")
# sc.addFile('./resources/dataheart_lexicon/negative_words.txt')
# sc.addFile("./resources/text_mining_lexicon/positive.txt")
# sc.addFile('./resources/text_mining_lexicon/negative.txt')
#
# dh_pos_adrs = SparkFiles.getRootDirectory()
# print("+++++++++++++++++++++++++++++ address: ", dh_pos_adrs)
# dh_neg_adrs = SparkFiles.get('negative_words.txt')
# tm_pos_adrs = SparkFiles.get('positive.txt')
# tm_neg_adrs = SparkFiles.get('negative.txt')