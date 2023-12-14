FROM python:3.10.12

RUN apt update -y && apt instapll awscli -y
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
