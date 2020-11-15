from __future__ import unicode_literals
from hazm import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Word2Vec
from pyspark.ml.classification import LinearSVC, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


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
    tokens = docs.rdd.flatMap(lambda doc: doc.text.split(' ')).toDF()
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
    high_conf = result_df.rdd\
        .filter(lambda x: x.probability[0] >= 0.8 or x.probability[1] >= 0.8).toDF()
    # high_conf.show(truncate=False)
    print(result_df.count(), "count of high confidense data", high_conf.count())
    evaluator = MulticlassClassificationEvaluator(labelCol="accept", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(result_df)
    print("Test set accuracy = " + str(accuracy))


def cross_validation(total_df):
    folds = 5
    print(folds, "fold cross validation")

    hashingTF = HashingTF(inputCol="tokens", outputCol="hashedTf", numFeatures=300)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="hashedTfIdf")
    rf = RandomForestClassifier(labelCol="accept", featuresCol=idf.getOutputCol(), predictionCol='prediction')
    svm = LinearSVC(labelCol='accept', featuresCol=idf.getOutputCol(), predictionCol='prediction')
    lgr = LogisticRegression(labelCol='accept', featuresCol=idf.getOutputCol(), predictionCol='prediction',
                             # maxIter=10, regParam=0.3, elasticNetParam=0.8
                             )

    pipeline = Pipeline(stages=[hashingTF, idf, lgr])

    param_grid = ParamGridBuilder().build()
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid,
                        evaluator=MulticlassClassificationEvaluator(labelCol="accept",
                                                                    predictionCol="prediction",
                                                                    metricName="accuracy"),
                        numFolds=folds, parallelism=2, seed=50)
    cv_model = cv.fit(total_df)
    print(cv_model.avgMetrics)


if __name__ == '__main__':
    sc = SparkContext(appName="Mobile")
    # sc.addPyFile(hazm)
    spark = SparkSession.builder.master("local[1]").appName("Mobile").getOrCreate()
    data_df = spark.read.csv('./dataset/mobile_digikala.csv', inferSchema=True, header=True)

    # get_info(data_df)
    data_df = tokenization(data_df)
    train, test = data_df.randomSplit([0.7, 0.3], seed=42)
    print("train and test count", train.count(), test.count())

    tfidf_train, tfidf_test = build_tfidf(train, test)
    w2v_train, w2v_test = build_word2vec(train, test)
    tfidf_train.printSchema()

    print("___________svm classifier with tf-idf embedding___________")
    svm_classification(tfidf_train, tfidf_test, feature_col='hashedTfIdf')
    print("___________svm classifier with word2vec embedding______________")
    svm_classification(w2v_train, w2v_test, feature_col='word2vec')

    print("___________RF classifier with tf-idf embedding___________")
    random_forest_classification(tfidf_train, tfidf_test, feature_col='hashedTfIdf')
    print("___________RF classifier with word2vec embedding______________")
    random_forest_classification(w2v_train, w2v_test, feature_col='word2vec')

    print("____________ cross validation ____________")
    cross_validation(data_df)
    spark.stop()

""" remained works: 
    1- hyper parameters tuning (random forest model, lgr, ...)
    2- implement using logistic regression
    3- hazm word_tokenization
"""