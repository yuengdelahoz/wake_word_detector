FROM tensorflow/tensorflow:latest-gpu
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y libsndfile1 
RUN pip3 install librosa
RUN pip3 install pydub
RUN apt install ffmpeg -y
RUN set -o vi
CMD ["bash"]

