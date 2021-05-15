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

def _prepare_data(AUDIO_FILE_PATH, OUTPUT_FOLDER, file_size_thresh = 100, min_highest_amplitude=-25):
	utils.create_folder(OUTPUT_FOLDER)
	cnt = 0
	exit = False
	for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
		try:
			if root == OUTPUT_FOLDER:
				continue
			for d in dirs:
				if 'MACOSX' in d:
					print ('deleting {}/{}'.format(root,d))
					shutil.rmtree(os.path.join(root,d))
			for f in filenames:
				file_path = os.path.join(root,f)
				size =math.ceil(os.path.getsize(file_path)/1024)
				if size <= file_size_thresh:
					try:
						audio_file = AudioSegment.from_file(file_path)
						if audio_file.dBFS < min_highest_amplitude:
							continue
					except KeyboardInterrupt:
						exit = True
						break
					except:
						audio_file = None

					if audio_file:
						duration = len(audio_file) 
						if duration <= 3000:
							start = int((3000 - duration)/2)
							new_audio = AudioSegment.silent(3000) 
							new_audio = new_audio.overlay(audio_file, position=start)
							new_audio = new_audio.set_channels(1)
							new_audio = new_audio.set_sample_width(2)
							new_audio = new_audio.set_frame_rate(16000)
							out_file = os.path.join(OUTPUT_FOLDER,'file_{:04d}.wav'.format(cnt))
							new_audio.export(out_file,format="wav")
							print(cnt,file_path, "duration: {} s, gain: {}".format(new_audio.duration_seconds, new_audio.dBFS))
							cnt += 1
			if exit:
				break
		except KeyboardInterrupt:
			break
		except:
			traceback.print_exc()
			pass

def prepare_mozilla_common_data(AUDIO_FILE_PATH):
	OUTPUT_FOLDER = "/audio_files/zero_class"
	_prepare_data(AUDIO_FILE_PATH, OUTPUT_FOLDER, 17)

def prepare_hey_ida_data(AUDIO_FILE_PATH):
	OUTPUT_FOLDER = "/audio_files/one_class"
	_prepare_data(AUDIO_FILE_PATH, OUTPUT_FOLDER, min_highest_amplitude = -100)
