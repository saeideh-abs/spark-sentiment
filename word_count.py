import os
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import os
from functools import reduce
from pyspark.sql import DataFrame
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"


def read_pandas_df():
    socialnet = 'news'
    path = '/media/sa/sa/uni&work/arshad/corona_competition/gitlab/phase2/data/all_data/' + socialnet + '_normal/'
    frames = []
    for i in range(1, 5):
        print("i:", i)
        df = pd.read_csv(path + socialnet + '_corona98' + str(i) + '_normalized_tokenized_pid.csv')
        frames.append(df)
    df = pd.concat(frames)
    df[['post_type', 'cluster_idx']] = df[['post_type', 'cluster_idx']].astype(str)
    print(df.head(20), df.columns)
    return df['textField_nlp_normal']


def read_spark_df():
    socialnet = 'news'
    path = '/media/sa/sa/uni&work/arshad/corona_competition/gitlab/phase2/data/all_data/' + socialnet + '_normal/'
    frames = []
    for i in range(1, 5):
        print("i:", i)
        df = spark.read.csv(path + socialnet + '_corona98' + str(i) + '_normalized_tokenized_pid.csv',
                            header=True, inferSchema=True)
        frames.append(df)

    df_complete = reduce(DataFrame.unionAll, frames)  # ro merge(concatenate) multiple dfs
    return df_complete


if __name__ == '__main__':
    print("hello spark")
    # sc = SparkContext(appName="PythonWordCount")
    spark = SparkSession.builder.master("local[1]").appName("PythonWordCount").getOrCreate()

    # lines = sc.textFile("./sample.txt", 1)
    df = read_spark_df()
    # df = spark.createDataFrame(pd_df)  # for converting pandas dataframe to spark dataframe
    docs_df = df.select('textField_nlp_normal')
    df.cache()
    df.printSchema()
    df.show(20)
    # pd_df = df.limit(5).toPandas()  # convert spark df to pandas df

    counts = df.rdd.filter(lambda arg: arg.textField_nlp_normal is not None) \
        .flatMap(lambda x: x.textField_nlp_normal.split(' ')) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(add)
    output = counts.collect()
    # for word, count in output:
    #     # print(line)
    #     print("%s: %i" % (word, count))
    print("total number of unique words", type(output), counts.count())
    spark.stop()
