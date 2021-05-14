FROM tensorflow/tensorflow:latest-gpu
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y libsndfile1 
RUN pip3 install librosa
RUN pip3 install pydub
RUN set -o vi
RUN apt install ffmpeg -y
WORKDIR /home/developer
CMD ["bash"]

