#! /bin/bash
#
# run.sh
# Copyright (C) 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
xhost +
docker run --rm -it \
	--net=host \
	--gpus all \
	-w /src \
	-v $DIR/src:/src \
	-v $DIR/audio_files:/audio_files \
	-v /home/yuengdelahoz/Music/mozilla_common_voice_01/clips/:/audio_files2 \
	tensorflow-gpu-audio

