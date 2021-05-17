#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
import os
from pydub import AudioSegment, silence, effects
from pydub.utils import make_chunks
import numpy as np

def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)


def generate_background_noise_clips(AUDIO_FILE_PATH, OUTPUT_FOLDER, split_every_x_seconds=3):
	create_folder(OUTPUT_FOLDER)
	exit = False
	cnt = 0
	for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
		if exit:
			break

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
				length = split_every_x_seconds * 1000 # this is in miliseconds
				chunks = make_chunks(audio_file, length)
				np.random.shuffle(chunks)
				chunks = chunks[0:100] # 5 minutes
				for clip in chunks:
					new_audio = clip.set_channels(1)
					new_audio = new_audio.set_sample_width(2)
					new_audio = new_audio.set_frame_rate(16000)
					out_file = os.path.join(OUTPUT_FOLDER,'file_{:04d}.wav'.format(cnt))
					print(file_path,'->',out_file)
					new_audio.export(out_file,format="wav")
					cnt += 1

