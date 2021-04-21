FROM public.ecr.aws/lambda/python:3.8

COPY ./Lexica \
    ./SpacyTranslation \
    ./LambdaHandler/index.py \
    requirements.txt \
    "./custom_spacy_models/en_core_web_lg-3.0.0.tar.gz" \
    ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "./en_core_web_lg-3.0.0.tar.gz"

CMD [ "index.handler" ]