
import os

import pandas

from sparknlp import start as startSession
import sparknlp.base as nlp
import sparknlp.annotator as annotators
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline

spark = startSession()
# spark = SparkSession.builder.appName("UpGoerWriter").getOrCreate()

## Setup Logging
logFile = "./runlog.txt"
if not os.path.exists(logFile):
    with open(logFile, 'w'):
        pass
logData = spark.read.text(logFile).cache()

documentAssembler = nlp.DocumentAssembler().setInputCol('text').setOutputCol('document').setCleanupMode('shrink')
sentenceDetector = annotators.SentenceDetector().setInputCols(['document']).setOutputCol('sentence')
tokenizer = annotators.Tokenizer().setInputCols(['sentence']).setOutputCol('word')
assembler = nlp.TokenAssembler().setInputCols(['sentence', 'word']).setOutputCol('assembled')
normalizer = annotators.Normalizer().setInputCols(['word']).setOutputCol('normal')
wordEmbeddings = annotators.BertEmbeddings.pretrained().setInputCols(['sentence', 'normal']).setOutputCol('embeddings')
sentenceEmbeddings = annotators.BertSentenceEmbeddings.pretrained().setInputCols(['sentence']).setOutputCol("sentence_embeddings")
finisher = nlp.Finisher().setInputCols(["token"]).setCleanAnnotations(False).setIncludeMetadata(True)

pipeline_model = nlp.PipelineModel(stages=[
    #explain_pipeline_model, finisher
    documentAssembler,
    sentenceDetector,
    tokenizer,
    finisher
])

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
pipeline_model = pipeline.fit(empty_df)

## Load sample data

# DataFrame approach
#data = spark.createDataFrame(sentences).toDF("text")
#model = pipeline.fit(sentences)
#annotations_finished_df = model.transform(data)
#annotations_finished_df.select('finished_token').show(truncate=False)

# Light Pipeline
with open('../tests/MobyDick_Chapter1.txt', 'r') as test:
    text = test.read()
output = nlp.LightPipeline(pipeline_model, parse_embeddings=True).annotate(text)
print(output['sentence'][0])
print(output['embeddings'][0])
print(output['sentence_embeddings'][0])

#df = spark.createDataFrame(pandas.DataFrame({'text': [text]}))
#result = pipeline_model.transform(df)
#result.show()
