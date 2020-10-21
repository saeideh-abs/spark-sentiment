from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkFiles

url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
sc = SparkContext()
sc.addFile(url)
spark = SparkSession.builder.appName("AdultApp").getOrCreate()
print("started")
df = spark.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema=True)
df.cache()
df.printSchema()
df.show(10, truncate=False)
df.groupBy("education").count().sort("count", ascending=True).show()
df.describe().show()
# df.drop('education_num').columns
df.filter(df.age > 40).count()