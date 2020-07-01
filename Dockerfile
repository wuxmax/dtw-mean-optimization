FROM tiangolo/python-machine-learning:python3.7

RUN mkdir /app

COPY ./app/requirements.txt /app
RUN conda install -y --file app/requirements.txt

COPY ./app /app

ENTRYPOINT [ "python3", "app/main.py" ]
