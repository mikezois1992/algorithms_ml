# Dockerfile, Image, Container
FROM python:3.12

ADD main.py .

RUN pip install streamlit
RUN pip install pandas
RUN pip install openpyxl

CMD ["python", "./main.py"]