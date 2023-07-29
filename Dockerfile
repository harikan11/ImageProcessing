FROM python:3
COPY . /app
WORKDIR /app
RUN apk update
RUN apk add py-pip
RUN apk add gfortran build-base openbias-dev libffi-dev
RUN pip3 install versioneer
RUN pip3 install cycler
RUN apk add --no-cache libpng-dev
RUN apk add --no-cache jpeg-dev zlib-dev
RUN apk add --no-cache --virtual .build-deps build-base linux-headers
RUN pip3 install Cython
RUN pip3 install Pillow
RUN pip3 install -r requirements.txt 
EXPOSE 5001 
ENTRYPOINT [ "python" ] 
CMD [ "trial.py" ] 