FROM python:3.8.18

COPY . /ml
WORKDIR /ml
RUN pip install -r requirements.txt

# keep the container running
ENTRYPOINT ["top", "-b"]