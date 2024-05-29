## Dockerfile, Image, Container
FROM python:3.12

WORKDIR /app

ADD . /app

# Install packages
RUN pip install streamlit pandas openpyxl matplotlib scikit-learn plotly

EXPOSE 8501

# Execute Streamlit
CMD ["streamlit", "run", "main.py"]

## Commands we run in Terminal
# docker build -t algorithms_ml .
# docker run -p 8501:8501 algorithms_ml