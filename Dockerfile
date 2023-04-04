# https://docs.docker.com/samples/library/python/
FROM python:3.9

WORKDIR /app

# basic installs
RUN apt-get update -y \
  && apt-get install -y swig \
  && apt-get install -y time

# copy essential stuff to image
COPY requirements.txt /app

# install project dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install gym[Box2D]
RUN pip3 install jupyter

# debugging tool - feel free to use it yourself! :)
RUN pip3 install snakeviz

COPY scripts/startup_script.sh /app/scripts/startup_script.sh
COPY rlcw /app/rlcw
COPY config.yml /app/config.yml

# give startup script permission to be ran as an executable
RUN chmod +x /app/scripts/startup_script.sh

ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["sh", "/app/scripts/startup_script.sh"]
