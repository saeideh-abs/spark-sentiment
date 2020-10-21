from __future__ import unicode_literals
import hazm
from hazm import *
from pyspark.sql import SparkSession
from pyspark import SparkContext


def get_info(df):
    df.printSchema()
    # df.show()
    labels = df.select('accept')
    comments = df.select('text')
    # comments.show(comments.count())
    # print(df.schema.names)
    positive_num = labels.filter(df.accept == 1).count()
    negative_num = labels.filter(df.accept == 0).count()
    print("number of positives:", positive_num, "number of negatives:", negative_num)
    comments.show()


def build_tfidf(docs):
    tokens = custom_tokenizer(docs)


def tokenize(doc):
    print(word_tokenize("sgh kjh al,"))
    return doc


def custom_tokenizer(docs):
    # docs.show(5)
    # new = docs.rdd.map(str)
    tokens = docs.rdd.map(lambda doc: tokenize(doc))
    tokens.take(5)


if __name__ == '__main__':
    sc = SparkContext(appName="Mobile")
    sc.addPyFile('./resources/hazm-master.zip')
    spark = SparkSession.builder.master("local[1]").appName("Mobile").getOrCreate()
    data_df = spark.read.csv('./dataset/mobile_digikala.csv', inferSchema=True, header=True)
    labels = data_df.select('accept')
    comments = data_df.select('text')
    # get_info(data_df)
    build_tfidf(comments)
