FROM pytorch/pytorch:latest

# The following line is required to install all dependencies for opencv
RUN apt update && apt install -y python-opencv

ADD backend/requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /backend

ADD backend/api.py .

ADD src ./src

ENV FLASK_APP api.py

# probably won't need to expose it since it will be accessed only by the
# frontend.
EXPOSE 5000

# Entrypont
CMD ["flask", "run", "--host", "0.0.0.0"]
