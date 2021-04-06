from __future__ import unicode_literals
import time
import glob
from pyspark.context import SparkConf, SparkContext
from functools import reduce
from pyspark.sql import DataFrame
from hazm import *
from pyspark.sql.functions import udf, split, concat_ws, when
from pyspark.sql import SparkSession
# from pyspark import SparkContext
from pyspark.sql.types import IntegerType, ArrayType, StringType
from pyspark.sql.functions import regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, HashingTF, IDF, Tokenizer, Word2Vec
from pyspark.ml.classification import LinearSVC, RandomForestClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def display_current_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


# a function for concatenate all crawled csv files and make a single csv file
def read_spark_df():
    all_files = glob.glob('./dataset/categories/*.csv')
    frames = []
    for filename in all_files:
        print(filename)
        df = spark.read.csv(filename, header=True, inferSchema=True)
        frames.append(df)

    df_complete = reduce(DataFrame.unionAll, frames)  # to merge(concatenate) multiple dfs
    df_complete.repartition(1).write.csv("digikalaCrawledData.csv", header=True)
    return df_complete


def miras_cleaning(df):
    # df = df.na.drop()
    # df = df.fillna(0)
    df = df.filter((df.accept == '-1') | (df.accept == '1') | (df.accept == '0'))
    print("count", df.count())
    # print(df.filter("accept is not NULL").count(), "count of null in accept")
    # print(df.filter("text is not NULL").count(), "count of null in text")
    df = df.withColumn('accept', regexp_replace('accept', '-1', '2'))
    df = df.withColumn("accept", df["accept"].cast(IntegerType()))

    positive_num = df.filter(df.accept == 1).count()
    neutral_num = df.filter(df.accept == 0).count()
    negative_num = df.filter(df.accept == 2).count()
    print("number of positives:", positive_num, "number of negatives:", negative_num,
          "number of neutrals:", neutral_num)
    return df


def digikala_crawled_cleaning(df):
    print("total number of data:", df.count())
    df = df.dropDuplicates(['product_id', 'holder', 'comment_title', 'comment_body'])
    print("number of data after removing duplicates:", df.count(), display_current_time())

    # df = df.withColumnRenamed('comment_body', 'text')
    df = df.withColumn('text', concat_ws(' ', df.comment_title, df.comment_body))

    # get some info
    df = df.filter((df.recommendation == 'opinion-positive') | (df.recommendation == 'opinion-negative') |
                   (df.recommendation == 'opinion-noidea'))
    print("count of labeled comments", df.count())
    print("positives count:", df.filter(df.recommendation == 'opinion-positive').count())
    print("negatives count:", df.filter(df.recommendation == 'opinion-negative').count())
    print("neutrals count:", df.filter(df.recommendation == 'opinion-noidea').count())

    df = df.rdd.filter(lambda arg: arg.text is not None).toDF()  # remove empty comments
    print("count of labled and non-empty comment_bodies:", df.count())
    # print("advantages", df.select('advantages').show(truncate=False))

    df = get_balance_samples(df)

    stringIndexer = StringIndexer(inputCol="recommendation", outputCol="accept", stringOrderType="frequencyDesc")
    model = stringIndexer.fit(df)
    df = model.transform(df)
    # # or
    # df = df.withColumn('accept', when(df.recommendation == 'opinion-positive', 1.0)
    #                    .when(df.recommendation == 'opinion-negative', 2.0)
    #                    .when(df.recommendation == 'opinion-noidea', 0.0)
    #                    )
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
    df.printSchema()
    # df.show()
    labels = df.select('accept')
    comments = df.select('text')
    print("count of comments", comments.count(), "count of labels:", labels.count())
    print("number of partiotions", df.rdd.getNumPartitions())


def tokenization(docs):
    # using spark tool
    # tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    # tokens = tokenizer.transform(docs)
    # or
    tokens = docs.withColumn('tokens', hazm_tokenizer('clean_text'))
    tokens.printSchema()
    return tokens


def custom_tokenizer(docs):
    # docs.show(5)
    # new = docs.rdd.map(str)
    # tokens = docs.rdd.flatMap(lambda doc: [doc.text.split(' ')])  # need to convert this to DF
    # tokens = docs.rdd.flatMap(lambda doc: [word_tokenize(str(doc.text))])
    # tokens = docs.withColumn("text", split("text", "\s+")).withColumnRenamed('text', 'tokens')
    tokens = docs.withColumn('tokens', hazm_tokenizer('clean_text'))
    # tokens.select('tokens').show(truncate=False)
    return tokens


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
    lemmatizer_udf = udf(lambda words: [lemmatizer.lemmatize(word).split('#')[0] for word in words], ArrayType(StringType()))
    conjoin_words = udf(lambda words_list: ' '.join(words_list), StringType())

    df = df.withColumn('normal_text', normalizer_udf('text'))
    df = df.withColumn('words', hazm_tokenizer('normal_text'))
    df = df.withColumn('without_stopwords', stopwords_remover('words'))
    # df = df.withColumn('stem', stemmer_udf('words'))
    # df = df.withColumn('lemm', lemmatizer_udf('without_stopwords'))  # because of negative verbs
    df = df.withColumn('clean_text', conjoin_words('without_stopwords'))
    # df.select('clean_text', 'normal_text').show(truncate=False)
    return df


def build_tfidf(train_df, test_df):
    hashingTF = HashingTF(inputCol="tokens", outputCol="hashedTf", numFeatures=300)
    train_featurizedData = hashingTF.transform(train_df)
    test_featurizedData = hashingTF.transform(test_df)

    # featurizedData.select('rawFeatures').show(1, truncate=False)

    idf = IDF(inputCol="hashedTf", outputCol="hashedTfIdf")
    idfModel = idf.fit(train_featurizedData)
    train_rescaledData = idfModel.transform(train_featurizedData)
    test_rescaledData = idfModel.transform(test_featurizedData)
    return train_rescaledData, test_rescaledData


def build_word2vec(train_df, test_df):
    word2vec = Word2Vec(vectorSize=300, minCount=5, inputCol='tokens', outputCol='word2vec')
    model = word2vec.fit(train_df)
    train_vec = model.transform(train_df)
    test_vec = model.transform(test_df)
    return train_vec, test_vec


def svm_classification(train_df, test_df, feature_col):
    svm = LinearSVC(featuresCol=feature_col, labelCol='accept')
    model = svm.fit(train_df)
    model.setPredictionCol("prediction")
    result_df = model.transform(test_df)
    # result_df.select('prediction').show()
    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("Test set accuracy = " + str(accuracy))


def random_forest_classification(train_df, test_df, feature_col):
    rf = RandomForestClassifier(labelCol="accept", featuresCol=feature_col, predictionCol='prediction')
    model = rf.fit(train_df)
    result_df = model.transform(test_df)
    # result_df.select('probability').show(truncate=False)
    # high_conf = result_df.rdd\
    #     .filter(lambda x: x.probability[0] >= 0.8 or x.probability[1] >= 0.8).toDF()
    # # high_conf.show(truncate=False)
    # print(result_df.count(), "count of high confidense data", high_conf.count())
    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("Test set accuracy = " + str(accuracy))


def naive_bayes_classification(train_df, test_df, feature_col):
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="accept", featuresCol=feature_col)
    model = nb.fit(train_df)
    result_df = model.transform(test_df)
    # result_df.select('prediction').show()
    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("Test set accuracy = " + str(accuracy))


def logistic_regression_classification(train_df, test_df, feature_col):
    lgr = LogisticRegression(labelCol='accept', featuresCol=feature_col, predictionCol='prediction',
                             # maxIter=10, regParam=0.3, elasticNetParam=0.8
                             )
    model = lgr.fit(train_df)
    result_df = model.transform(test_df)
    # result_df.select('prediction').show()
    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("Test set accuracy = " + str(accuracy))


def cross_validation(total_df):
    print("number of partiotions", total_df.rdd.getNumPartitions())
    folds = 5
    print(folds, "fold cross validation")

    # hashingTF = HashingTF(inputCol="tokens", outputCol="hashedTf", numFeatures=300)
    # idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="hashedTfIdf")
    # print("hashing tf was finished")

    word2vec = Word2Vec(vectorSize=300, minCount=5, inputCol='tokens', outputCol='word2vec')
    print("word2vec")

    # rf = RandomForestClassifier(labelCol="accept", featuresCol=idf.getOutputCol(), predictionCol='prediction')
    # svm = LinearSVC(labelCol='accept', featuresCol=idf.getOutputCol(), predictionCol='prediction')
    lgr = LogisticRegression(labelCol='accept', featuresCol=word2vec.getOutputCol(), predictionCol='prediction',
                             # maxIter=10, regParam=0.3, elasticNetParam=0.8
                             )
    # pipeline = Pipeline(stages=[hashingTF, idf, lgr])
    pipeline = Pipeline(stages=[word2vec, lgr])
    param_grid = ParamGridBuilder().build()
    print("param grid")
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid,
                        evaluator=MulticlassClassificationEvaluator(labelCol="accept",
                                                                    predictionCol="prediction",
                                                                    metricName="accuracy"),
                        numFolds=folds, parallelism=4, seed=50)
    total_df.cache()
    print("cv model fit")
    cv_model = cv.fit(total_df)
    print(cv_model.avgMetrics)


if __name__ == '__main__':
    print("start time:", display_current_time())

    # _______________________ spark configs _________________________
    # master address: "spark://master:7077"
    conf = SparkConf().setMaster("local[*]").setAppName("digikala comments sentiment")
    spark_context = SparkContext(conf=conf)

    spark = SparkSession(spark_context).builder.master("local[*]")\
        .appName("digikala comments sentiment")\
        .getOrCreate()
    # print("spark", spark.master)
    print("****************************************")

    # _______________________ loading datasets _________________________
    # data_df = spark.read.csv('./dataset/3000ـmobile_digikala.csv', inferSchema=True, header=True)

    # data_df = spark.read.csv('./dataset/miras_opinion.csv', inferSchema=True, header=True)
    # data_df = miras_cleaning(data_df)

    # data_df = spark.read.csv('./dataset/digikala_all.csv', inferSchema=True, header=True)
    data_df = spark.read.csv('hdfs://master:9000/user/saeideh/digikala_all.csv', inferSchema=True, header=True)
    print("data was loaded from hdfs", display_current_time())
    # data_df = data_df.limit(2300000)

    print(spark_context.defaultParallelism)
    data_df = data_df.repartition(spark_context.defaultParallelism)
    print("number of partitions:", data_df.rdd.getNumPartitions())
    data_df = digikala_crawled_cleaning(data_df)
    data_df = data_df.repartition(spark_context.defaultParallelism)
    print("number of partitions:", data_df.rdd.getNumPartitions())
    get_info(data_df)

    # ____________________ preprocessing and embedding _____________________
    data_df = data_df.select('text', 'accept')
    print("text cleaner func", display_current_time())
    data_df = text_cleaner(data_df)

    print("tokenizer", display_current_time())
    data_df = tokenization(data_df)
    # data_df.select('tokens').show(truncate=False)
    train, test = data_df.randomSplit([0.7, 0.3], seed=42)
    # print("train and test count", train.count(), test.count(), display_current_time())

    print("tf-idf embedding", display_current_time())
    tfidf_train, tfidf_test = build_tfidf(train, test)
    # print("word2vec embedding", display_current_time())
    # w2v_train, w2v_test = build_word2vec(train, test)
    # tfidf_train.printSchema()

    # _____________________ classification part _______________________
    print("___________svm classifier with tf-idf embedding___________", display_current_time())
    # svm_classification(tfidf_train, tfidf_test, feature_col='hashedTfIdf')
    print("___________svm classifier with word2vec embedding______________", display_current_time())
    # svm_classification(w2v_train, w2v_test, feature_col='word2vec')

    print("___________RF classifier with tf-idf embedding___________", display_current_time())
    # random_forest_classification(tfidf_train, tfidf_test, feature_col='hashedTfIdf')
    print("___________RF classifier with word2vec embedding______________", display_current_time())
    # random_forest_classification(w2v_train, w2v_test, feature_col='word2vec')

    print("___________NB classifier with tf-idf embedding___________", display_current_time())
    naive_bayes_classification(tfidf_train, tfidf_test, feature_col='hashedTfIdf')
    print("___________NB classifier with word2vec embedding______________", display_current_time())
    # naive_bayes_classification(w2v_train, w2v_test, feature_col='word2vec')

    print("___________lgr classifier with tf-idf embedding___________", display_current_time())
    # logistic_regression_classification(tfidf_train, tfidf_test, feature_col='hashedTfIdf')
    print("___________lgr classifier with word2vec embedding______________", display_current_time())
    # logistic_regression_classification(w2v_train, w2v_test, feature_col='word2vec')

    print("____________ cross validation ____________", display_current_time())
    # cross_validation(data_df)
    print("end time:", display_current_time())
    spark.stop()

""" remained works:
    1- hyper parameters tuning (random forest model, lgr, ...)
    2- lexicon based and hybrid clf
    3- preprocessing (punctuation removing)
    4- bert and parsbert
    
    ./start-slave.sh spark://172.23.178.8:7077
    spark-submit --master spark://172.23.178.8:7077  file.py
    start-dfs.sh
    master:8080  //spark running apps
    master:4040 //spark jobs, storage ,...
    master:50070 , master:9000, sa-master:9870 //hadoop hdfs
"""