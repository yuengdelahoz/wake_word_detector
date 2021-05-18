#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
from . import utils
from pydub import AudioSegment
import os

import pydub
from python_speech_features import mfcc
import numpy as np

def _generate_mfcc(audio_file):
	audio_clip = np.frombuffer(audio_file.get_array_of_samples(), dtype=np.int16)
	return mfcc(audio_clip, samplerate=audio_file.frame_rate)

def _generate_dataset(AUDIO_FILE_PATH, OUTPUT_FOLDER):
	utils.create_folder(OUTPUT_FOLDER)
	exit = False
	global_cnt = 0
	for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
		for f in filenames:
			file_path = os.path.join(root,f)
			try:
				audio_file = AudioSegment.from_file(file_path)
			except KeyboardInterrupt:
				exit = True
				break
			except:
				audio_file = None

			if audio_file:
				mfcc_feat = _generate_mfcc(audio_file)
				if mfcc_feat.shape != (299,13):
					print(global_cnt,"wrong shape",file_path,mfcc_feat.shape, len(audio_file))
				global_cnt += 1

				if global_cnt % 300 == 0:
					print(global_cnt ,file_path,mfcc_feat.shape)


			if exit:
				break

def generate_dataset():
	AUDIO_FILE_PATH = "/audio_files/dglobal_cnt ataset/classes"
	OUTPUT_FOLDER = "/audio_files/dataset"
	_generate_dataset(AUDIO_FILE_PATH,OUTPUT_FOLDER)
