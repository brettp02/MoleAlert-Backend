FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#ENV PYTHONPATH "${PYTHONPATH}:/code/app/"

COPY ./app/ /code/app/
COPY ./app/best_model.pth /code/best_model.pth


CMD ["fastapi","run","app/main.py","--port","80"]
