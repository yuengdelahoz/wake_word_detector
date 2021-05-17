#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""

from data import prepare_mozilla_common_data, prepare_hey_ida_data
from data import utils 

prepare_hey_ida_data("/audio_files/originals/test")
# prepare_mozilla_common_data("/audio_files2")

# utils.generate_background_noise_clips("/audio_files/originals/background_noise","/audio_files/background_noise_chunks")

# import librosa

# y, sr = librosa.load("/audio_files/originals/joe/1.m4a")
# print(len(y),sr)

# librosa.output.write_wav('test.wav', y, sr)

