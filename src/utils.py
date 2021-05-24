#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
import time
import os
from python_speech_features import mfcc
import librosa
import numpy as np
from pydub.utils import make_chunks
from pydub import AudioSegment

def timing_val(func):
	def wrapper(*arg, **kw):
		t1 = time.time()*1000
		res = func(*arg, **kw)
		t2 = time.time()*1000
		print('{} took {:.2f} ms'.format(func.__name__,(t2 - t1)))
		return res
	return wrapper

def timing_val2(func):
	def wrapper(*arg, **kw):
		t1 = time.time()*1000
		res = func(*arg, **kw)
		t2 = time.time()*1000
		duration = '{} took {:.2f} ms'.format(func.__name__,(t2 - t1))
		return res,duration
	return wrapper

def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

@timing_val
def generate_mfcc(audio_signal, sample_rate=16000, n_mfcc=20, n_mels=26, n_fft=2048, hop_length=512, use="psf"):
	if isinstance(audio_signal,AudioSegment):
		audio_samples= audio_signal.get_array_of_samples()
		audio_array = np.frombuffer(audio_signal, dtype=np.int16)
	elif isinstance(audio_signal, np.array):
		audio_array = audio_signal
	elif isinstance(audio_signal,bytes):

	if use == "librosa":
		return librosa.feature.mfcc(y=audio_signal,
				n_mfcc=n_mfcc,
				n_fft=n_fft,
				sr=sample_rate,
				hop_length=hop_length
				).transpose() # required to get it in the right dimensions
	elif use == "psf":

		return mfcc(audio_array,
				nfft=n_fft,
				samplerate=sample_rate,
				winlen=n_fft/sample_rate,
				winstep=hop_length/sample_rate,
				appendEnergy=False,
				winfun=np.hanning
				)

def split_audio_in_chunks(audio_file, chunk_len_in_millis= 100, mfcc_lib = "psf"):
	if mfcc_lib == "librosa":
		pass
	else: # psf
		chunks = make_chunks(audio_file,chunk_len_in_millis)
