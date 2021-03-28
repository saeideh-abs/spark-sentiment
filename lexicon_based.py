from __future__ import unicode_literals
import polarity_determination as polde
import time
from hazm import *
from pyspark.context import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, udf, regexp_replace, when
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def display_current_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


def digikala_crawled_cleaning(df):
    print("total number of data:", df.count(), display_current_time())
    df = df.dropDuplicates(['product_id', 'holder', 'comment_title', 'comment_body'])
    print("number of data after removing duplicates:", df.count(), display_current_time())

    # df = df.withColumnRenamed('comment_body', 'text')
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
                       .when(df.recommendation == 'opinion-negative', -1.0)
                       .when(df.recommendation == 'opinion-noidea', 0.0))

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
    balance_df = reduce(DataFrame.unionAll, [balance_neg, balance_pos])
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


def predict_polarities(df):
    print("entered in lexicon based method", display_current_time())

    text_polarity_udf = udf(polde.text_polarity, DoubleType())
    result_df = df.withColumn('prediction', text_polarity_udf('clean_text'))
    result_df.select('accept', 'prediction').show(50, truncate=False)
    print("lexicon based polarity ditection was finished", display_current_time())

    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("lexicon based method accuracy = " + str(accuracy), display_current_time())
    # tp_rate_1 = evaluator.evaluate(result_df, {evaluator.metricName: "truePositiveRateByLabel",
    #                              evaluator.metricLabel: 1.0})
    # fp_rate_1 = evaluator.evaluate(result_df, {evaluator.metricName: "falsePositiveRateByLabel",
    #                                            evaluator.metricLabel: 1.0})
    # print("tp_rate_1,  fp_rate_1:", tp_rate_1, fp_rate_1, display_current_time())
    # tp_rate_neg = evaluator.evaluate(result_df, {evaluator.metricName: "truePositiveRateByLabel",
    #                                            evaluator.metricLabel: -1.0})
    # fp_rate_neg = evaluator.evaluate(result_df, {evaluator.metricName: "falsePositiveRateByLabel",
    #                                            evaluator.metricLabel: -1.0})
    # print("tp_rate_neg,  fp_rate_neg:", tp_rate_neg, fp_rate_neg, display_current_time())
    return result_df


def binary_confusion_matrix(df, target_col, prediction_col):
    print("binary confusion matrix", display_current_time())
    tp = df[(df[target_col] == 1) & (df[prediction_col] == 1)].count()
    tn = df[(df[target_col] == -1) & (df[prediction_col] == -1)].count()
    fp = df[(df[target_col] == -1) & (df[prediction_col] == 1)].count()
    fn = df[(df[target_col] == 1) & (df[prediction_col] == -1)].count()

    print("tp    tn    fp    fn", display_current_time())
    print(tp, tn, fp, fn)

    tnu = df[(df[target_col] == 0) & (df[prediction_col] == 0)].count()
    fnup = df[(df[target_col] == 1) & (df[prediction_col] == 0)].count()
    fnun = df[(df[target_col] == -1) & (df[prediction_col] == 0)].count()

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
    data_df = spark.read.csv('./dataset/digikala_all.csv', inferSchema=True, header=True)
    # data_df = data_df.limit(100000)
    # data_df = spark.read.csv('hdfs://master:9000/user/saeideh/digikala_all.csv', inferSchema=True, header=True)
    print("data was loaded from hdfs", display_current_time())

    data_df = digikala_crawled_cleaning(data_df)

    get_info(data_df)

    # ____________________ preprocessing _____________________
    train, data_df = data_df.randomSplit([0.7, 0.3], seed=42)

    data_df = data_df.select('text', 'accept')
    print("text cleaner func", display_current_time())
    data_df = text_cleaner(data_df)

    # ____________________ classification using lexicons _____________________
    new_df = predict_polarities(data_df)
    binary_confusion_matrix(new_df, target_col='accept', prediction_col='prediction')

    print("end time:", display_current_time())
    spark.stop()
