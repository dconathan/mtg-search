FROM public.ecr.aws/lambda/python:3.8

# RUN pip install -v torch transformers fastapi pydantic mangum requests pytorch-lightning

COPY dist/* .

RUN pip install mtg_search-0.2.0-py3-none-any.whl

CMD ["mtg_search.app.app.handler"]
