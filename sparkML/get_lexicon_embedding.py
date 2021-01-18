import os
import json

import pandas

from sparknlp import start as startSession
import sparknlp.base as nlp
import sparknlp.annotator as annotators
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline


def create_lexicon_embedding(lexicon):
    spark = startSession()
    
    pipeline = Pipeline(stages=[
        nlp.DocumentAssembler().setInputCol('text').setOutputCol('document').setCleanupMode('shrink'),
        annotators.Tokenizer().setInputCols(['document']).setOutputCol('word'),
        annotators.Normalizer().setInputCols(['word']).setOutputCol('normal'),
        annotators.BertEmbeddings.pretrained().setInputCols(['document', 'normal']).setOutputCol('embedding'),
        nlp.EmbeddingsFinisher().setInputCols(["embedding"]).setOutputCols(['embedding_vector']).setCleanAnnotations(True),
    ])

    empty_df = spark.createDataFrame([['']], ['text'])
    pipeline_model = pipeline.fit(empty_df)

    ## Load sample data
    with open(lexicon, 'r') as lexica_io:
        words = lexica_io.readlines()

    # Light Pipeline
    in = spark.createDataFrame(pandas.DataFrame({'text': words}))
    output = nlp.LightPipeline(pipeline_model).transform(in).select('*').toPandas()
    print(output)
    embedding_lookup = {row['text'].strip(): row['embedding_vector'][0] for _, row in output.iterrows()}

    
    lexicon_filename, _ = os.path.splitext(lexicon)
    with open(lexicon_filename + '_BERTembeddings.json', 'w') as embeddings_io:
        json.dump(embedding_lookup, embeddings_io)


if __name__ == '__main__':
    create_lexicon_embedding('../Lexica/most_common_1000.txt')
