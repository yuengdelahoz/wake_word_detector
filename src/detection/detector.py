#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
import pyaudio
import numpy as np
# from .engine import Engine
from python_speech_features import mfcc
import time
from threading import Thread, Event

def timing_val(func):
	def wrapper(*arg, **kw):
		t1 = time.time()*1000
		f = func(*arg, **kw)
		t2 = time.time()*1000
		print('It took {:.2f} ms to execute function {}'.format((t2 - t1), func.__name__))
		return f
	return wrapper

class Detector:
	def __init__(self, source='mic'):
		self.source = source
	
	def start(self):
		if self.source == 'mic':
			self.chunk_size = 1600
			sample_format = pyaudio.paInt16 
			channels = 1
			self.frame_rate = 16000
			self.seconds = 3

			p = pyaudio.PyAudio() 
			self.stream = p.open(format=sample_format,
					channels=channels,
					rate=self.frame_rate,
					frames_per_buffer=self.chunk_size,
					input=True)

			self.running = True
			self.is_paused = False
			self.thread = Thread(target=self._handle_predictions, daemon=True)
			self.thread.daemon = True
			self.thread.start()


	# @timing_val
	def _generate_mfcc(self,bytestring ,fs):
		audio_clip = np.frombuffer(bytestring, dtype=np.int16)
		return mfcc(audio_clip, fs)

	@timing_val
	def _get_next_bytestring(self,frames, num_of_required_frames):
		frames.append(self.stream.read(self.chunk_size))
		if len(frames) <  num_of_required_frames:
			return
		elif len(frames) > num_of_required_frames:
			while len(frames) != num_of_required_frames:
				frames.pop(0)

		if len(frames) == num_of_required_frames:
			bytestring = b''.join(frames)
			mfcc_feat = self._generate_mfcc(bytestring,self.frame_rate)
			# print('_handle_predictions',mfcc_feat.shape)
		

	def _handle_predictions(self):
		print('Recording')
		num_of_required_frames = self.frame_rate * self.seconds/ self.chunk_size
		frames = list()
		while self.running:
			self._get_next_bytestring(frames,num_of_required_frames)
