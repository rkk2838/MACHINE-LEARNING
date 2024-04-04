FROM python:3.8-slim-buster 

COPY ./requirements.txt requirements.txt

COPY ./main.py main.py 

# RUN pip --version
RUN pip install -r requirements.txt
ENV NAME Docker_venv
CMD ["python","./main.py"]


