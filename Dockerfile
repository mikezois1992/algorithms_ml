# Dockerfile, Image, Container
FROM python:3.12

ADD main.py .

RUN pip install streamlit

CMD ["python", "./main.py"]