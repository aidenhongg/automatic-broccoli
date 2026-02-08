FROM python:3.11-slim


RUN apt-get update && apt-get install -y awscli
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu 
RUN python3 -m pip install numpy

WORKDIR /app

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]