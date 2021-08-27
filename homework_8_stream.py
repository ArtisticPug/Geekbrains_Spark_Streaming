# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

import json
from pyspark.sql.types import StructType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import PipelineModel

features = ['PassengerId', 'Pclass', 'Sex_index',
            'Age', 'SibSp', 'Parch',
            'Fare', 'Embarked_index']
text_columns = ['Sex', 'Embarked']

spark = SparkSession.builder.appName("mllib_predict_app").getOrCreate()
checkpoint_location = "tmp/ml_checkpoint"
# upload the best model
data_path = "lesson_8/data_source/titanic/*"
model_dir = "lesson_8/models"
model = PipelineModel.load(model_dir + "/ml_model_titanic")
# read stream
schema = StructType.fromJson(json.loads('{"fields":[{"metadata":{},"name":"PassengerId","nullable":true,"type":"integer"},{"metadata":{},"name":"Survived","nullable":true,"type":"integer"},{"metadata":{},"name":"Pclass","nullable":true,"type":"integer"},{"metadata":{},"name":"Name","nullable":true,"type":"string"},{"metadata":{},"name":"Sex","nullable":true,"type":"string"},{"metadata":{},"name":"Age","nullable":true,"type":"double"},{"metadata":{},"name":"SibSp","nullable":true,"type":"integer"},{"metadata":{},"name":"Parch","nullable":true,"type":"integer"},{"metadata":{},"name":"Ticket","nullable":true,"type":"string"},{"metadata":{},"name":"Fare","nullable":true,"type":"double"},{"metadata":{},"name":"Cabin","nullable":true,"type":"string"},{"metadata":{},"name":"Embarked","nullable":true,"type":"string"}],"type":"struct"}'))
data = spark \
    .readStream \
    .format("csv") \
    .schema(schema) \
    .options(header=True, maxFilesPerTrigge=1) \
    .load(data_path)


def filter_data(data, null_list):
    result_data = data
    for item in null_list:
        result_data = result_data.filter("%s is not null" % item)
    return result_data


def null_value_calc(df):
    null_columns_counts = []
    numrows = df.count()
    for k in df.columns:
        nullrows = df.where(col(k).isNull()).count()
        if nullrows > 0:
            temp = k, nullrows, (nullrows / numrows) * 100
            null_columns_counts.append(temp)
    return null_columns_counts


def prepare_data(df, features, text_columns):
    data_is_clean = False
    null_columns_calc_list = null_value_calc(df)
    if null_columns_calc_list:
        spark.createDataFrame(null_columns_calc_list, ['Column_Name', 'Null_Values_Count', 'Null_Value_Percent'])
    else:
        data_is_clean = True
        print("Data is clean with no null values")
    if data_is_clean is not True:
        null_tuples_list = null_value_calc(df)
        null_list = []
        for el in null_tuples_list:
            null_list.append(el[0])
        filtered_data = filter_data(df, null_list)
    else:
        filtered_data = df
    model_data = filtered_data
    for c in text_columns:
        string_indexer = StringIndexer(inputCol=c, outputCol=c + '_index').setHandleInvalid("keep")
        model_data = string_indexer.fit(model_data).transform(model_data)
    model_data_transformed = model_data.select(features)
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    model_data_result = assembler.transform(model_data_transformed)
    return model_data_result.select('features')


def process_batch(df, epoch):
    model_data = prepare_data(df, features, text_columns)
    prediction = model.transform(model_data)
    prediction.show()


def foreach_batch_output(df):
    from datetime import datetime as dt
    date = dt.now().strftime("%Y%m%d%H%M%S")
    return df \
        .writeStream \
        .trigger(processingTime='%s seconds' % 10) \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", checkpoint_location + "/" + date) \
        .start()


stream = foreach_batch_output(data)

stream.awaitTermination()

stream.stop()