FROM public.ecr.aws/lambda/python:3.8

RUN pip install -v torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers fastapi pydantic mangum pytorch-lightning

COPY dist/* .

RUN pip install *.whl

CMD ["mtg_search.api.app.handler"]
