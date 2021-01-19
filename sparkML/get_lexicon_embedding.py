import os
import json

import pandas

from sparknlp import start as startSession
import sparknlp.base as nlp
import sparknlp.annotator as annotators
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline


LEXICON_DIR = '../Lexica/'


def create_lexicon_embedding(lexicon):
    spark = startSession()
    
    pipeline = Pipeline(stages=[
        nlp.DocumentAssembler().setCleanupMode('shrink')\
           .setInputCol('text').setOutputCol('document'),
        annotators.Tokenizer()\
            .setInputCols(['document']).setOutputCol('word'),
        annotators.Normalizer()\
            .setInputCols(['word']).setOutputCol('normal'),
        annotators.BertEmbeddings.pretrained()\
            .setInputCols(['document', 'normal']).setOutputCol('embedding'),
        nlp.EmbeddingsFinisher().setCleanAnnotations(False)\
            .setInputCols(["embedding"]).setOutputCols(['embedding_vector']),
    ])

    empty_df = spark.createDataFrame([['']], ['text'])
    pipeline_model = pipeline.fit(empty_df)

    ## Load sample data
    with open(os.path.join(LEXICON_DIR, lexicon + '.txt'), 'r') as lexica_io:
        words = lexica_io.readlines()

    # Light Pipeline
    in_frame = spark.createDataFrame(pandas.DataFrame({'text': words}))
    out_frame = nlp.LightPipeline(pipeline_model).transform(in_frame).select('*').toPandas()
    embedding_lookup = {
        row['text'].strip(): row['embedding_vector'][0]
        for _, row in out_frame.iterrows()
    }

    
    lexicon_filename, _ = os.path.splitext(lexicon)
    with open(os.path.join(LEXICON_DIR, lexicon + '_BERTembeddings.json'), 'w') as embeddings_io:
        json.dump(embedding_lookup, embeddings_io)


def load_lexicon(lexicon):
    with open(os.path.join(LEXICON_DIR, lexicon + '_BERTembeddings.json')) as lexicon_io:
        return json.load(lexicon_io)


if __name__ == '__main__':
    create_lexicon_embedding('../Lexica/most_common_1000.txt')
