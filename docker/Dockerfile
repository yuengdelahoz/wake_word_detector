FROM tensorflow/tensorflow:latest-gpu
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y libsndfile1 
RUN pip3 install librosa
RUN pip3 install pydub
RUN apt install ffmpeg -y
RUN apt install libgstreamer1.0-0 -y
RUN pip3 install PyGObject
RUN pip3 install python_speech_features
RUN set -o vi
CMD ["bash"]

