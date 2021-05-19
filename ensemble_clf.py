from __future__ import unicode_literals
import polarity_determination as polde
import time
import ast
from hazm import *
from pyspark.context import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, udf, regexp_replace, when, array, concat
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType
from pyspark.ml.feature import Word2Vec, HashingTF, IDF
from pyspark.ml.linalg import Vectors, VectorUDT
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
    balance_df = reduce(DataFrame.unionAll, [balance_pos, balance_neg, balance_neut])
    return balance_df


def get_info(df):
    print("*********** get_info func *************")
    # df.printSchema()
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
    # tokens.printSchema()
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
    hashingTF = HashingTF(inputCol="tokens", outputCol="hashedTf", numFeatures=15000)
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


def lexicon_features(df):
    print("entered in lexicon based method", display_current_time())

    text_polarity_udf = udf(polde.extract_features, ArrayType(IntegerType()))
    result_df = df.withColumn('lexicon_features', text_polarity_udf('clean_text', 'advantages', 'disadvantages'))
    print("lexicon based feature extraction was finished", display_current_time())
    return result_df


def logistic_regression_classification(train_df, test_df, feature_col):
    print("entered in logistic regression clf", display_current_time())
    lgr = LogisticRegression(labelCol='accept', featuresCol=feature_col, predictionCol='lgr_prediction',
                             # maxIter=10, regParam=0.3, elasticNetParam=0.8
                             )
    model = lgr.fit(train_df)
    train_result_df = model.transform(train_df)
    test_result_df = model.transform(test_df)
    test_result_df = test_result_df.withColumnRenamed('rawPrediction', 'lgrRawPrediction')\
        .withColumnRenamed('probability', 'lgrProbability')
    train_result_df = train_result_df.withColumnRenamed('rawPrediction', 'lgrRawPrediction') \
        .withColumnRenamed('probability', 'lgrProbability')

    # test_result_df.printSchema()
    # binary_confusion_matrix(test_result_df, 'accept', 'lgr_prediction')
    evaluation(test_result_df, 'accept', 'lgr_prediction', "LGR")
    return train_result_df, test_result_df


def naive_bayes_classification(train_df, test_df, feature_col):
    print("entered in naive bayes clf", display_current_time())
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="accept", featuresCol=feature_col,
                    predictionCol='nb_prediction')
    model = nb.fit(train_df)
    train_result_df = model.transform(train_df)
    test_result_df = model.transform(test_df)
    # test_result_df.printSchema()
    test_result_df = test_result_df.withColumnRenamed('rawPrediction', 'nbRawPrediction')\
        .withColumnRenamed('probability', 'nbProbability')
    train_result_df = train_result_df.withColumnRenamed('rawPrediction', 'nbRawPrediction') \
        .withColumnRenamed('probability', 'nbProbability')
    # binary_confusion_matrix(test_result_df, 'accept', 'nb_prediction')
    evaluation(test_result_df, 'accept', 'nb_prediction', "NB")
    return train_result_df, test_result_df


def random_forest_classification(train_df, test_df, feature_col):
    print("entered in rf clf", display_current_time())
    rf = RandomForestClassifier(labelCol="accept", featuresCol=feature_col, predictionCol='rf_prediction')
    model = rf.fit(train_df)

    train_result_df = model.transform(train_df)
    test_result_df = model.transform(test_df)
    test_result_df = test_result_df.withColumnRenamed('rawPrediction', 'rfRawPrediction')\
        .withColumnRenamed('probability', 'rfProbability')
    train_result_df = train_result_df.withColumnRenamed('rawPrediction', 'rfRawPrediction') \
        .withColumnRenamed('probability', 'rfProbability')

    # test_result_df.printSchema()
    # binary_confusion_matrix(test_result_df, 'accept', 'rf_prediction')
    evaluation(test_result_df, 'accept', 'rf_prediction', "RF")
    return train_result_df, test_result_df


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


def merge_features(df):
    print("entered in merge_features func ", display_current_time())
    df = df.withColumn('ml_features', array('lgr_prediction', 'nb_prediction'))
    to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
    df = df.withColumn('merged_features', to_vector(concat(df.ml_features, df.lexicon_features)))
    return df


def list2vector(df):
    print("entered in list2vector func ", display_current_time())
    to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
    df = df.withColumn('merged_features', to_vector(df.lexicon_features))
    return df


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


def evaluation(df, target_col, prediction_col, classifier_name):
    evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol=prediction_col,
                                                  metricName="accuracy", metricLabel=1.0)
    accuracy = evaluator.evaluate(df)
    print(classifier_name + " Test set accuracy = " + str(accuracy), display_current_time())
    # evaluator.setMetricName('f1')
    # f1 = evaluator.evaluate(df)
    # print("f1:", f1, display_current_time())
    #
    evaluator.setMetricName('weightedFMeasure')
    Wfmeasure = evaluator.evaluate(df)
    print("weighted f measure:", Wfmeasure, display_current_time())

    evaluator.setMetricName('weightedPrecision')
    Wprecision = evaluator.evaluate(df)
    print("weightedPrecision:", Wprecision, display_current_time())

    evaluator.setMetricName('weightedRecall')
    Wrecall = evaluator.evaluate(df)
    print("weightedRecall:", Wrecall, display_current_time())

    # evaluator.setMetricName('precisionByLabel')
    # precision2 = evaluator.evaluate(df)
    # print("PrecisionByLabel:", precision2, display_current_time())
    #
    # evaluator.setMetricName('recallByLabel')
    # recall2 = evaluator.evaluate(df)
    # print("recallByLabel:", recall2, display_current_time())
    #
    # evaluator.setMetricName('fMeasureByLabel')
    # fmeasure2 = evaluator.evaluate(df)
    # print("fMeasureByLabel:", fmeasure2, display_current_time())


if __name__ == '__main__':
    print("start time:", display_current_time())

    # _______________________ spark configs _________________________
    conf = SparkConf().setMaster("spark://master:7077").setAppName("digikala comments sentiment, ensemble")
    spark_context = SparkContext(conf=conf)
    spark_context.addPyFile("./polarity_determination.py")

    spark = SparkSession(spark_context).builder.master("spark://master:7077") \
        .appName("digikala comments sentiment, ensemble clf") \
        .getOrCreate()
    print("****************************************")

    # _______________________ loading dataset _________________________
    data_df = spark.read.csv('hdfs://master:9000/user/saeideh/digikala_dataset.csv', inferSchema=True, header=True)
    print("data was loaded from hdfs", display_current_time())

    # data_df = data_df.limit(960000)
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
                                      seed=15
                                      )
    print("train and test count", train.count(), test.count(), display_current_time())

    # ____________________ classification part _____________________
    # # method 1: do soft voting between classifiers predicts
    # tfidf_train, tfidf_test = build_tfidf(train, test)
    # w2v_train, w2v_test = build_word2vec(tfidf_train, tfidf_test)
    #
    # result_df = lexicon_based(w2v_test)
    # # result_df = logistic_regression_classification(w2v_train, result_df, feature_col='word2vec')
    # result_df = naive_bayes_classification(tfidf_train, result_df, feature_col='hashedTfIdf')
    # print("number of partitions: ", data_df.rdd.getNumPartitions())
    # result_df = random_forest_classification(w2v_train, result_df, feature_col='word2vec')
    # result_df = soft_voting(result_df)
    # # result_df.select('accept', 'ensemble_prediction', 'lexicon_prediction', 'lgr_prediction', 'rf_prediction')\
    # #     .show(50, truncate=False)
    # print("end time:", display_current_time())

    # # method 2: get classifiers predictions as features
    tfidf_train, tfidf_test = build_tfidf(train, test)
    # w2v_train, w2v_test = build_word2vec(tfidf_train, tfidf_test)

    train_result_df = lexicon_features(tfidf_train)
    test_result_df = lexicon_features(tfidf_test)

    train_result_df, test_result_df = logistic_regression_classification(train_result_df, test_result_df, feature_col='hashedTfIdf')
    train_result_df, test_result_df = naive_bayes_classification(train_result_df, test_result_df, feature_col='hashedTfIdf')
    print("number of partitions: ", data_df.rdd.getNumPartitions())
    # train_result_df, test_result_df = random_forest_classification(train_result_df, test_result_df, feature_col='hashedTfIdf')

    train_result_df = merge_features(train_result_df)
    test_result_df = merge_features(test_result_df)
    # # or
    # train_result_df = list2vector(train_result_df) # for just using lexicon features
    # test_result_df = list2vector(test_result_df)

    # test_result_df.select('accept', 'lgr_prediction', 'rf_prediction', 'nb_prediction', 'merged_features')\
    #     .show(50, truncate=False)

    # ensemble features
    train_result_df = train_result_df.select('merged_features', 'accept')
    test_result_df = test_result_df.select('merged_features', 'accept')

    train_result_df = train_result_df.repartition(spark_context.defaultParallelism)
    test_result_df = test_result_df.repartition(spark_context.defaultParallelism)
    print("number of partiotions", train_result_df.rdd.getNumPartitions(), test_result_df.rdd.getNumPartitions())

    result_df = logistic_regression_classification(train_result_df, test_result_df, feature_col='merged_features')
    print("end time:", display_current_time())

    spark.stop()
