from __future__ import unicode_literals
import polarity_determination as polde
import time
import ast
from hazm import *
from pyspark.context import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, udf, regexp_replace, when
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType, DoubleType
from pyspark.ml.feature import Word2Vec, HashingTF, IDF
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def display_current_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


@udf(returnType=StringType())
def list2string_udf(list):
    list = ast.literal_eval(list)
    # str = ' '.join(list)
    str = ''
    for item in list:
        str = str + ' ' + item.strip('\r')

    return str


def digikala_crawled_cleaning(df):
    print("total number of data:", df.count(), display_current_time())
    df = df.dropDuplicates(['product_id', 'holder', 'comment_title', 'comment_body'])
    print("number of data after removing duplicates:", df.count(), display_current_time())

    # df = df.withColumnRenamed('comment_body', 'text')

    # df = df.withColumn('advantages_str', list2string_udf(df.advantages))
    # df = df.withColumn('disadvantages_str', list2string_udf(df.disadvantages))
    # df = df.withColumn('text',
    #                    concat_ws(' ', df.comment_title, df.comment_body, df.advantages_str, df.disadvantages_str))

    df = df.withColumn('text', concat_ws(' ', df.comment_title, df.comment_body))

    # get some info
    df = df.filter((df.recommendation == 'opinion-positive') | (df.recommendation == 'opinion-negative') |
                   (df.recommendation == 'opinion-noidea')
                   )
    print("count of labeled comments", df.count())
    print("positives count:", df.filter(df.recommendation == 'opinion-positive').count())
    print("negatives count:", df.filter(df.recommendation == 'opinion-negative').count())
    print("neutrals count:", df.filter(df.recommendation == 'opinion-noidea').count())

    df = df.rdd.filter(lambda arg: arg.text is not None).toDF()  # remove empty comments
    print("count of labled and non-empty comment_bodies:", df.count())

    df = get_balance_samples(df)

    # stringIndexer = StringIndexer(inputCol="recommendation", outputCol="accept", stringOrderType="frequencyDesc")
    # model = stringIndexer.fit(df)
    # df = model.transform(df)
    # # or
    df = df.withColumn('accept', when(df.recommendation == 'opinion-positive', 1.0)
                       .when(df.recommendation == 'opinion-negative', 0.0)
                       .when(df.recommendation == 'opinion-noidea', 2.0))

    # print(df.select('accept', 'recommendation').show(50, truncate=False))
    return df


def get_balance_samples(df):
    print("entered in get_balance_samples func", display_current_time())

    positive_df = df.filter((df.recommendation == 'opinion-positive'))
    negative_df = df.filter((df.recommendation == 'opinion-negative'))
    neutral_df = df.filter((df.recommendation == 'opinion-noidea'))

    pos_count = positive_df.count()
    neg_count = negative_df.count()
    neut_count = neutral_df.count()

    min_count = min(pos_count, neg_count, neut_count)
    print("positive comments:", pos_count, "negative comments:", neg_count,
          "neutral comments:", neut_count)
    print("min count = ", min_count)

    balance_pos = positive_df.limit(min_count)
    balance_neg = negative_df.limit(min_count)
    balance_neut = neutral_df.limit(min_count)

    print("balance positive comments:", balance_pos.count(), "balance negative comments:", balance_neg.count(),
          "balance neutral comments:", balance_neut.count())
    balance_df = reduce(DataFrame.unionAll, [balance_pos, balance_neg])
    return balance_df


def get_info(df):
    print("*********** get_info func *************")
    df.printSchema()
    # df.show()
    labels = df.select('accept')
    comments = df.select('text')
    print("count of comments", comments.count(), "count of labels:", labels.count())
    print("number of partiotions", df.rdd.getNumPartitions())


@udf(returnType=ArrayType(StringType()))
def hazm_tokenizer(text):
    return word_tokenize(text)


# preprocessing on persian text
def text_cleaner(df):
    normalizer = Normalizer(persian_numbers=False)
    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    stopwords = open('./resources/stopwords.txt', encoding="utf8").read().split('\n')

    normalizer_udf = udf(normalizer.normalize, StringType())
    stopwords_remover = udf(lambda words: [word for word in words if word not in stopwords], ArrayType(StringType()))
    stemmer_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
    lemmatizer_udf = udf(lambda words: [lemmatizer.lemmatize(word).split('#')[0] for word in words],
                         ArrayType(StringType()))
    conjoin_words = udf(lambda words_list: ' '.join(words_list), StringType())

    df = df.withColumn('normal_text', normalizer_udf('text'))
    df = df.withColumn('words', hazm_tokenizer('normal_text'))
    df = df.withColumn('without_stopwords', stopwords_remover('words'))
    # df = df.withColumn('stem', stemmer_udf('words'))
    # df = df.withColumn('lemm', lemmatizer_udf('without_stopwords'))  # because of negative verbs
    df = df.withColumn('clean_text', conjoin_words('without_stopwords'))
    # df.select('clean_text', 'normal_text').show(truncate=False)
    return df


def tokenization(docs):
    tokens = docs.withColumn('tokens', hazm_tokenizer('clean_text'))
    tokens.printSchema()
    return tokens


def build_word2vec(train_df, test_df):
    print("entered in build_word2vec fun", display_current_time())
    word2vec = Word2Vec(vectorSize=300, minCount=5, inputCol='tokens', outputCol='word2vec')
    model = word2vec.fit(train_df)
    train_vec = model.transform(train_df)
    test_vec = model.transform(test_df)
    return train_vec, test_vec


def build_tfidf(train_df, test_df):
    print("entered in build_tfidf fun", display_current_time())
    hashingTF = HashingTF(inputCol="tokens", outputCol="hashedTf", numFeatures=300)
    train_featurizedData = hashingTF.transform(train_df)
    test_featurizedData = hashingTF.transform(test_df)

    # featurizedData.select('rawFeatures').show(1, truncate=False)

    idf = IDF(inputCol="hashedTf", outputCol="hashedTfIdf")
    idfModel = idf.fit(train_featurizedData)
    train_rescaledData = idfModel.transform(train_featurizedData)
    test_rescaledData = idfModel.transform(test_featurizedData)
    return train_rescaledData, test_rescaledData


def lexicon_based(df):
    print("entered in lexicon based method", display_current_time())

    text_polarity_udf = udf(polde.find_label, DoubleType())
    result_df = df.withColumn('lexicon_prediction', text_polarity_udf('clean_text', 'advantages', 'disadvantages'))
    print("lexicon based polarity ditection was finished", display_current_time())

    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="lexicon_prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("lexicon based method accuracy = " + str(accuracy), display_current_time())
    return result_df


def logistic_regression_classification(train_df, test_df, feature_col):
    print("entered in logistic regression clf", display_current_time())
    lgr = LogisticRegression(labelCol='accept', featuresCol=feature_col, predictionCol='lgr_prediction',
                             # maxIter=10, regParam=0.3, elasticNetParam=0.8
                             )
    model = lgr.fit(train_df)
    result_df = model.transform(test_df)
    result_df = result_df.withColumnRenamed('rawPrediction', 'lgrRawPrediction')\
        .withColumnRenamed('probability', 'lgrProbability')

    # result_df.printSchema()

    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="lgr_prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("LGR Test set accuracy = " + str(accuracy), display_current_time())
    # binary_confusion_matrix(result_df, 'accept', 'lgr_prediction')
    return result_df


def naive_bayes_classification(train_df, test_df, feature_col):
    print("entered in naive bayes clf", display_current_time())
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="accept", featuresCol=feature_col,
                    predictionCol='nb_prediction')
    model = nb.fit(train_df)
    result_df = model.transform(test_df)
    # result_df.printSchema()
    result_df = result_df.withColumnRenamed('rawPrediction', 'nbRawPrediction')\
        .withColumnRenamed('probability', 'nbProbability')

    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="nb_prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("NB Test set accuracy = " + str(accuracy), display_current_time())
    # result_df.select('accept', 'nb_prediction', 'lexicon_prediction').show(50, truncate=False)
    # binary_confusion_matrix(result_df, 'accept', 'nb_prediction')
    return result_df


def random_forest_classification(train_df, test_df, feature_col):
    print("entered in rf clf", display_current_time())
    rf = RandomForestClassifier(labelCol="accept", featuresCol=feature_col, predictionCol='rf_prediction')
    model = rf.fit(train_df)
    result_df = model.transform(test_df)
    result_df = result_df.withColumnRenamed('rawPrediction', 'rfRawPrediction')\
        .withColumnRenamed('probability', 'rfProbability')

    # result_df.printSchema()

    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="rf_prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("RF Test set accuracy = " + str(accuracy))
    # binary_confusion_matrix(result_df, 'accept', 'rf_prediction')
    return result_df


@udf(returnType=DoubleType())
def ensemble_predict_udf(lexicon_label, ml1_label, ml2_label, ml1Probability, ml2Probability):
    if ml1_label == lexicon_label and ml1_label == ml2_label:
        label = ml1_label
    else:
        ml2_label_prob = max(ml2Probability)
        ml1_label_prob = max(ml1Probability)
        if ml2_label_prob > ml1_label_prob:
            label = ml2_label
        else:
            label = ml1_label
    return label


def soft_voting(df):
    print("entered in soft voting ", display_current_time())
    result_df = df.withColumn('ensemble_prediction', ensemble_predict_udf(
        'lexicon_prediction', 'nb_prediction', 'rf_prediction', 'nbProbability', 'rfProbability'))

    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="ensemble_prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("ensemble clf Test set accuracy = " + str(accuracy), display_current_time())
    binary_confusion_matrix(result_df, 'accept', 'ensemble_prediction')
    return result_df


def binary_confusion_matrix(df, target_col, prediction_col):
    print("binary confusion matrix", display_current_time())
    tp = df[(df[target_col] == 1) & (df[prediction_col] == 1)].count()
    tn = df[(df[target_col] == 0) & (df[prediction_col] == 0)].count()
    fp = df[(df[target_col] == 0) & (df[prediction_col] == 1)].count()
    fn = df[(df[target_col] == 1) & (df[prediction_col] == 0)].count()

    print("tp    tn    fp    fn", display_current_time())
    print(tp, tn, fp, fn)

    tnu = df[(df[target_col] == 2) & (df[prediction_col] == 2)].count()
    fnup = df[(df[target_col] == 1) & (df[prediction_col] == 2)].count()
    fnun = df[(df[target_col] == 0) & (df[prediction_col] == 2)].count()

    print("tnu       fnup       fnun")
    print(tnu, fnup, fnun)


if __name__ == '__main__':
    print("start time:", display_current_time())

    # _______________________ spark configs _________________________
    conf = SparkConf().setMaster("local[*]").setAppName("digikala comments sentiment, lexicon based")
    spark_context = SparkContext(conf=conf)

    spark = SparkSession(spark_context).builder.master("local[*]") \
        .appName("digikala comments sentiment, lexicon based") \
        .getOrCreate()
    print("****************************************")

    # _______________________ loading dataset _________________________
    data_df = spark.read.csv('hdfs://master:9000/user/saeideh/digikala_all.csv', inferSchema=True, header=True)
    print("data was loaded from hdfs", display_current_time())

    # data_df = data_df.limit(100000)
    data_df = data_df.repartition(spark_context.defaultParallelism)
    data_df = digikala_crawled_cleaning(data_df)
    data_df = data_df.repartition(spark_context.defaultParallelism)
    get_info(data_df)

    # ____________________ preprocessing _____________________
    data_df = data_df.select('text', 'accept', 'advantages', 'disadvantages')
    print("text cleaner func", display_current_time())
    data_df = text_cleaner(data_df)
    print("tokenizer", display_current_time())
    data_df = tokenization(data_df)

    train, test = data_df.randomSplit([0.7, 0.3],
                                      seed=42
                                      )
    print("train and test count", train.count(), test.count(), display_current_time())

    # ____________________ classification part _____________________
    tfidf_train, tfidf_test = build_tfidf(train, test)
    w2v_train, w2v_test = build_word2vec(tfidf_train, tfidf_test)

    result_df = lexicon_based(w2v_test)
    # result_df = logistic_regression_classification(w2v_train, result_df, feature_col='word2vec')
    result_df = naive_bayes_classification(tfidf_train, result_df, feature_col='hashedTfIdf')
    print("number of partitions: ", data_df.rdd.getNumPartitions())
    result_df = random_forest_classification(w2v_train, result_df, feature_col='word2vec')
    result_df = soft_voting(result_df)
    # result_df.select('accept', 'ensemble_prediction', 'lexicon_prediction', 'lgr_prediction', 'rf_prediction')\
    #     .show(50, truncate=False)
    print("end time:", display_current_time())
    spark.stop()
