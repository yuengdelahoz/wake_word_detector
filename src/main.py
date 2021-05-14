#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
import librosa
import os
import shutil
import utils
import numpy as np
from pydub import AudioSegment, silence
import math
import traceback
import sys

# AUDIO_FILE_PATH = "/audio_files/nate_hey_ida2"
AUDIO_FILE_PATH = "/audio_files/"
# AUDIO_FILE_PATH= "/audio_files/wav_files"
OUTPUT_FOLDER = "/audio_files/wav_files"
utils.create_folder(OUTPUT_FOLDER)
cnt = 0
for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
	try:
		if root == AUDIO_FILE_PATH:
			continue
		if root == OUTPUT_FOLDER:
			continue
		for d in dirs:
			if 'MACOSX' in d:
				print ('deleting {}/{}'.format(root,d))
				shutil.rmtree(os.path.join(root,d))
		for f in filenames:
			file_path = os.path.join(root,f)
			audio_file = AudioSegment.from_file(file_path)
			duration = len(audio_file) 

			if duration <= 3000:
				# print(audio_file.frame_rate, audio_file.channels, audio_file.sample_width)
				start = int((3000 - duration)/2)
				new_audio = AudioSegment.silent(3000) 
				new_audio = new_audio.overlay(audio_file, position=start)
				new_audio = new_audio.set_channels(1)
				new_audio = new_audio.set_sample_width(2)
				new_audio = new_audio.set_frame_rate(16000)
				# print(new_audio.frame_rate, new_audio.channels, new_audio.sample_width)
				out_file = os.path.join(OUTPUT_FOLDER,'file_{:04d}.wav'.format(cnt))
				new_audio.export(out_file,format="wav")
				print(cnt,file_path, "processed")
				cnt += 1
	except KeyboardInterrupt:
		break
	except:
		traceback.print_exc()
		pass

print("DONE")
