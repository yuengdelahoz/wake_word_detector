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

def _fix_duration_and_convert_audio(AUDIO_FILE_PATH, OUTPUT_FOLDER, duration_thres = 3000, file_size_thresh = 100, min_highest_amplitude=-25):
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
						if duration <= duration_thres :
							start = int((duration_thres - duration)/2)
							new_audio = AudioSegment.silent(duration_thres) 
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
	OUTPUT_FOLDER = "/audio_files/common_voice_corpus_1"
	exists = os.path.exists(OUTPUT_FOLDER)
	if not exists  or ( exists and len(os.listdir(OUTPUT_FOLDER)) <= 20000):
		_fix_duration_and_convert_audio(AUDIO_FILE_PATH, OUTPUT_FOLDER, file_size_thresh = 17)

	audio_clips = os.listdir(OUTPUT_FOLDER) # total clips
	OUTPUT_FOLDER_2 ="/audio_files/zero_class"

	exists = os.path.exists(OUTPUT_FOLDER_2)
	if not exists  or ( exists and len(os.listdir(OUTPUT_FOLDER_2)) != 10000):
		utils.create_folder(OUTPUT_FOLDER_2 )

		#Choose 10000 audio clips randomly
		np.random.shuffle(audio_clips)
		audio_clips = audio_clips[:9999]

		#save audio clips 
		for aud_clip in audio_clips:
			src_file = os.path.join(OUTPUT_FOLDER, aud_clip) 
			dst_file = os.path.join(OUTPUT_FOLDER_2, aud_clip) 
			shutil.copyfile(src_file, dst_file)
			print('copying',src_file,dst_file)
	

def prepare_hey_ida_data(AUDIO_FILE_PATH):
	OUTPUT_FOLDER = "/audio_files/one_class"
	_fix_duration_and_convert_audio(AUDIO_FILE_PATH, OUTPUT_FOLDER, min_highest_amplitude = -100)
