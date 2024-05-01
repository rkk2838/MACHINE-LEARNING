FROM python:3.8-slim-buster
MAINTAINER Ravikant Kumar
WORKDIR /app
COPY . .
EXPOSE 8501
ENV NAME test_env
RUN pip install -r requirements.txt

# CMD ["python3", "app.py","streamlit run app.py"]
CMD ["streamlit", "run", "app.py"]




