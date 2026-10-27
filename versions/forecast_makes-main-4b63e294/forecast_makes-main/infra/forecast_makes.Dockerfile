FROM python:3.10-slim-bullseye

ENV  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_HOME="/home/poetry" \
  VENV_PATH="/home/forecast_makes .venv" \
  USER="forecast_makes"

ENV  PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN useradd $USER

RUN apt-get update \
#  && apt-get install -y git build-essential locales ffmpeg curl \
  && pip3 install --upgrade pip \
  && pip3 install poetry==1.4.2

WORKDIR /home/$USER
#ENV path_src="src/$USER"
ENV path_src="src"
#RUN export path_src="src/$USER"
#COPY src/$USER/pyproject.toml ./
RUN sed -i '/pt_BR.UTF-8/s/^# //g' /etc/locale.gen && \
  locale-gen
ENV LANG pt_BR.UTF-8
ENV LANGUAGE pt_BR:en
ENV LC_ALL pt_BR.UTF-8
RUN locale

COPY $path_src/pyproject.toml ./
#COPY $path_src/requirements.txt ./
RUN poetry run pip3 install -U pip setuptools
#RUN poetry config virtualenvs.create false
RUN poetry install 
#RUN poetry run pip3 install -r requirements.txt

#RUN mkdir files
#RUN mkdir files/anexos

#COPY $USER/app app
#COPY $path_src/$USER/app app
#COPY $path_src/migrations /home/$USER/migrations
#COPY $path_src/$USER/run.py $path_src/$USER/config.py $path_src/$USER/boot.sh ./
COPY $path_src/* .
COPY .env ./
RUN chmod +x boot.sh

ENV FLASK_APP run.py

RUN chown -R $USER:$USER /home/$USER
USER $USER

ENTRYPOINT ["sh", "./boot.sh"]
