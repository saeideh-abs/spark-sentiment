from __future__ import unicode_literals
from hazm import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
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


def build_tfidf(docs):
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    tokens = tokenizer.transform(docs)
    # tokens = custom_tokenizer(docs)
    hashingTF = HashingTF(inputCol="tokens", outputCol="hashedTf", numFeatures=300)
    featurizedData = hashingTF.transform(tokens)
    # featurizedData.select('rawFeatures').show(1, truncate=False)

    idf = IDF(inputCol="hashedTf", outputCol="hashedTfIdf")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    return rescaledData


def custom_tokenizer(docs):
    # docs.show(5)
    # new = docs.rdd.map(str)
    tokens = docs.rdd.map(lambda doc: doc.text.split(' ')).toDF()
    return tokens


def svm_classification(train_df, test_df):
    svm = LinearSVC(featuresCol='hashedTfIdf', labelCol='accept')
    model = svm.fit(train_df)
    model.setPredictionCol("prediction")
    result_df = model.transform(test)
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
    featurized_df = build_tfidf(data_df)
    featurized_df.printSchema()
    train, test = featurized_df.randomSplit([0.9, 0.1])
    print("train and test count", train.count(), test.count())
    svm_classification(train, test)
    spark.stop()
