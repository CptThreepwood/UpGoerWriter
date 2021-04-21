import os
import boto3

from SpacyTranslation import Translator

translator = Translator('en_core_web_lg', 'most_common_1000')

def handler(event, context):
    input_sentence = event['input']
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'test/plain',
        },
        'body': translator.translate(input_sentence),
    }

if __name__ == "__main__":
    EVENT = {'input': 'To make a thief, make an owner; to create crime, create laws'}
    CONTEXT = {}
    handler(EVENT, CONTEXT)