FROM public.ecr.aws/lambda/python:3.8

COPY . ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "./models/spacy/en_core_web_lg-3.0.0.tar.gz"

CMD [ "handler.handler" ]