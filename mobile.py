from __future__ import unicode_literals
from hazm import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Word2Vec
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


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


def tokenization(docs):
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    tokens = tokenizer.transform(docs)
    # tokens = custom_tokenizer(docs)
    return tokens


def custom_tokenizer(docs):
    # docs.show(5)
    # new = docs.rdd.map(str)
    tokens = docs.rdd.map(lambda doc: doc.text.split(' ')).toDF()
    return tokens


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
    word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol='tokens', outputCol='word2vec')
    model = word2vec.fit(train_df)
    train_vec = model.transform(train_df)
    test_vec = model.transform(test_df)
    train_vec.show(5, truncate=False)
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


if __name__ == '__main__':
    sc = SparkContext(appName="Mobile")
    # sc.addPyFile(hazm)
    spark = SparkSession.builder.master("local[1]").appName("Mobile").getOrCreate()
    data_df = spark.read.csv('./dataset/mobile_digikala.csv', inferSchema=True, header=True)

    # get_info(data_df)
    data_df = tokenization(data_df)
    train, test = data_df.randomSplit([0.8, 0.2])
    print("train and test count", train.count(), test.count())
    train_featurized, test_featurized = build_tfidf(train, test)
    train_featurized, test_featurized = build_word2vec(train, test)

    train_featurized.printSchema()
    svm_classification(train_featurized, test_featurized, feature_col='word2vec')
    spark.stop()
