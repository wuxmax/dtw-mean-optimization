FROM tiangolo/python-machine-learning:python3.7

# just to show cpu usage
RUN apt-get update
RUN apt-get install -y sysstat

RUN mkdir /app

COPY ./app/requirements.txt /app
RUN conda install -y --file app/requirements.txt

COPY ./app /app

ENTRYPOINT [ "python3", "app/main.py" ]
