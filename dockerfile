FROM python:3.8-slim
COPY . /app
WORKDIR /app
# RUN apk update
# RUN apt install espeak
# RUN apt install libespeak-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE $PORT 
EXPOSE 3000
CMD python /app/app.py