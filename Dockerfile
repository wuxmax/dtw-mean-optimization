# FROM nvidia/cuda:10.2-runtime-ubuntu18.04
FROM tiangolo/python-machine-learning:cuda9.1-python3.7

# RUN mkdir /results
# RUN chown -R /results

RUN mkdir /app
WORKDIR /app

COPY ./app/requirements.txt /app
RUN conda install -y --file requirements.txt

COPY ./app /app

# ENTRYPOINT [ "bash" ]
# CMD ["scripts/run_experiment.sh"]

ENTRYPOINT [ "python3", "main.py" ]
# CMD [ "main.py" ]
