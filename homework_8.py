# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString, VectorIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("mllib_prepare_and_train_app").master("local[*]").getOrCreate()

# ==================================================

data_path = "lesson_8/data_source/titanic/titanic.csv"
model_dir = "lesson_8/models"

# read data from storage
data = spark \
    .read \
    .format("csv") \
    .options(inferSchema=True, header=True) \
    .load(data_path)

print(data.schema.json())

data = data.select(col('PassengerId'), col('Survived'), col('Pclass'), col('Sex'), col('Age'), col('SibSp'),
                   col('Parch'), col('Fare'), col('Embarked'))

print(data.schema.json())

data_is_clean = False


# Used this function found here:
# https://towardsdatascience.com/heart-disease-prediction-using-apache-spark-ml-808073f52495
# to find the rows that needed cleaning


def null_value_calc(df):
    null_columns_counts = []
    numrows = df.count()
    for k in df.columns:
        nullrows = df.where(col(k).isNull()).count()
        if nullrows > 0:
            temp = k, nullrows, (nullrows / numrows) * 100
            null_columns_counts.append(temp)
    return null_columns_counts


def filter_data(data, null_list):
    result_data = data
    for item in null_list:
        result_data = result_data.filter("%s is not null" % item)
    return result_data


null_columns_calc_list = null_value_calc(data)
if null_columns_calc_list:
    spark.createDataFrame(null_columns_calc_list, ['Column_Name', 'Null_Values_Count', 'Null_Value_Percent']).show()
else:
    data_is_clean = True
    print("Data is clean with no null values")

if data_is_clean is not True:
    null_tuples_list = null_value_calc(data)
    null_list = []
    for el in null_tuples_list:
        null_list.append(el[0])
    filtered_data = filter_data(data, null_list)
    filtered_data.show(10)
else:
    filtered_data = data

# ==================================================

text_columns = ['Sex', 'Embarked']
output_text_columns = [c + "_index" for c in text_columns]

model_data = filtered_data

for c in text_columns:
    string_indexer = StringIndexer(inputCol=c, outputCol=c + '_index').setHandleInvalid("keep")
    model_data = string_indexer.fit(model_data).transform(model_data)

model_data.show(10)

# ==================================================

target = ['Survived']

features = ['PassengerId', 'Pclass', 'Sex_index',
            'Age', 'SibSp', 'Parch',
            'Fare', 'Embarked_index']

all_columns = ",".join(features + target).split(",")
model_data_transformed = model_data.select(all_columns)

model_data_transformed.show(10)

# ==================================================

assembler = VectorAssembler(inputCols=features, outputCol='features')
model_data_result = assembler.transform(model_data_transformed)
model_data_result = model_data_result.select('features', col(target[0]).alias('label'))

model_data_result.show(10)

# ==================================================

train, test = model_data_result.randomSplit([0.8, 0.2], seed=12345)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(model_data_result)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = \
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(model_data_result)

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(train)

# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(30)

predictions_transformed = predictions.select(col("predictedLabel").cast('double'), "label", "features")

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="predictedLabel", metricName="accuracy")

evaluator2 = RegressionEvaluator() \
    .setMetricName("rmse") \
    .setLabelCol("label") \
    .setPredictionCol("predictedLabel")

accuracy = evaluator.evaluate(predictions_transformed)
accuracy2 = evaluator2.evaluate(predictions_transformed)

print("Evaluator 1: Test Error = %g" % (1.0 - accuracy))
print("Evaluator 2: Test Error = %g" % accuracy2)

model.write().overwrite().save(model_dir + "/ml_model_titanic")
