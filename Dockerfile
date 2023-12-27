# The builder image, used to build the virtual environment
FROM python:3.11-slim-buster

RUN apt-get update

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY CodeHelperAgent.py CodeHelperAgent.py
COPY ./.chainlit ./.chainlit
COPY chainlit.md chainlit.md
COPY .env .env

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8000
EXPOSE 8000

CMD ["chainlit", "run", "CodeHelperAgent.py"]

