# Dockerfile, Image, Container
FROM python:3.12

ADD main.py .

RUN pip install streamlit
RUN pip install pandas
RUN pip install openpyxl
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install plotly

CMD ["python", "./main.py"]