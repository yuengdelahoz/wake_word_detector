#mfcc! /usr/bin/env python3
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
from python_speech_features import mfcc
import time
from threading import Thread
from .audio_streams import MicrophoneStream
from .engine import Engine

class Detector:
	def __init__(self, source='mic'):
		self.source = source
		self.frames = list()
		self.engine = Engine()
	
	def start(self, callback):
		if self.source == 'mic':
			self.mic_stream =  MicrophoneStream(self)
			self.num_of_required_frames = self.mic_stream.start()

			self.running = True
			self.thread = Thread(target=self._handle_predictions, args=(callback,))
			self.thread.start()

			print("Listening to microphone ...")
			while True:
				try:
					time.sleep(1)
				except KeyboardInterrupt:
					self.mic_stream.stop()
					self.stop()
					break

	def stop(self):
		if self.thread:
			self.running = False
			self.thread.join()
			self.thread = None
			print('Done listening')


	def _generate_mfcc(self,bytestring):
		audio_clip = np.frombuffer(bytestring, dtype=np.int16)
		return mfcc(audio_clip, self.mic_stream.frame_rate)

	def add_audio_chunk(self,chunk):
		self.frames.append(chunk)

	def _handle_predictions(self, callback):
		while self.running:
			if self.mic_stream.new_chunk_event.is_set():
				if len(self.frames) <  self.num_of_required_frames:
					pass
				elif len(self.frames) > self.num_of_required_frames:
					while len(self.frames) != self.num_of_required_frames:
						self.frames.pop(0)

				if len(self.frames) == self.num_of_required_frames:
					bytestring = b''.join(self.frames)
					mfcc_feat = self._generate_mfcc(bytestring)
					pred = self.engine.predict(mfcc_feat)
					if callback:
						callback(pred)
				print('frames',len(self.frames))

				self.mic_stream.new_chunk_event.clear()
