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
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import librosa
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import time

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
				k = 600 # audio clips counter
				for clip in chunks:
					if clip.duration_seconds != split_every_x_seconds: # ignore audio clips that are not split_every_x_seconds in duration
						continue
					k -= 1
					new_audio = clip.set_channels(1)
					new_audio = new_audio.set_sample_width(2)
					new_audio = new_audio.set_frame_rate(16000)
					out_file = os.path.join(OUTPUT_FOLDER,'file_{:04d}.wav'.format(cnt))
					print(file_path,'->',out_file, len(new_audio))
					new_audio.export(out_file,format="wav")
					cnt += 1
					if k < 0:
						break

def audiosegment_to_librosawav(audiosegment):    
	channel_sounds = audiosegment.split_to_mono()
	samples = [s.get_array_of_samples() for s in channel_sounds]

	fp_arr = np.array(samples).T.astype(np.float32)
	fp_arr /= np.iinfo(samples[0].typecode).max
	fp_arr = fp_arr.reshape(-1)

	return fp_arr

def _generate_mfcc(audio_file):
	audio_clip = np.frombuffer(audio_file.get_array_of_samples(), dtype=np.int16)
	return mfcc(audio_clip, samplerate=audio_file.frame_rate)


def timing_val(func):
	def wrapper(*arg, **kw):
		t1 = time.time()
		res = func(*arg, **kw)
		t2 = time.time()
		print((t2 - t1), res, func.__name__)
	return wrapper

@timing_val
def example():
	path = '/audio_files/test/yueng_file_00000000.wav'
	sr = 16000
	# audio_file = AudioSegment.from_file(path)
	# mfcc_psf1 = _generate_mfcc(audio_file)
	# print('mfcc_psf1 ',mfcc_psf1[0])

	# y1, sr = librosa.load(path, sr=sr)
	# mfcc_librosa = librosa.feature.mfcc(y=y1, sr=sr,n_mfcc=13)
	# y2 = audiosegment_to_librosawav(audio_file)
	# diff = y1 - y2
	# for d in diff:
		# print(d)
	# print('mfcc_psf1.shape',mfcc_psf1.shape)
	# print('mfcc_psf2.shape',mfcc_psf2.shape)


	# (rate,sig) = wav.read(path)
	# mfcc_psf2 = mfcc(sig,rate)
	# print('mfcc_psf2 ',mfcc_psf2[0])

