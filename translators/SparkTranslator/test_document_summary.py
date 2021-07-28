
import os

import pandas

from sparknlp import start as startSession
import sparknlp.base as nlp
import sparknlp.annotator as annotators
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline

spark = startSession()

## Setup Logging
logFile = "./runlog.txt"
if not os.path.exists(logFile):
    with open(logFile, 'w'):
        pass
logData = spark.read.text(logFile).cache()

documentAssembler = nlp.DocumentAssembler().setCleanupMode('shrink')\
    .setInputCol('text').setOutputCol('document')
sentenceDetector = annotators.SentenceDetector().setExplodeSentences(True)\
    .setInputCols(['document']).setOutputCol('sentence')
tokenizer = annotators.Tokenizer()\
    .setInputCols(['sentence']).setOutputCol('word')
assembler = nlp.TokenAssembler()\
    .setInputCols(['sentence', 'word']).setOutputCol('assembled')
normalizer = annotators.Normalizer()\
    .setInputCols(['word']).setOutputCol('normal')
wordEmbeddings = annotators.BertEmbeddings.pretrained()\
    .setInputCols(['sentence', 'normal']).setOutputCol('embeddings')
sentenceEmbeddings = annotators.BertSentenceEmbeddings.pretrained()\
    .setInputCols(['sentence']).setOutputCol("sentence_embeddings")
finisher = nlp.Finisher().setCleanAnnotations(False).setIncludeMetadata(True)\
    .setInputCols(["token"])

pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    normalizer,
    #assembler,
    wordEmbeddings,
    sentenceEmbeddings,
])
empty_df = spark.createDataFrame([['']]).toDF('text')
model = pipeline.fit(empty_df)

## Load sample data

# DataFrame approach
with open('../tests/MobyDick_Chapter1.txt', 'r') as test:
    document = test.read()
data = spark.createDataFrame(pandas.DataFrame({'text': [document]}))
annotations_finished_df = model.transform(data)
annotations_finished_df.select('*').show(truncate=True)
pandasDF = annotations_finished_df.select('*').toPandas()

import pickle
pickle.dump(pandasDF, open('test.pickle', 'wb'))

