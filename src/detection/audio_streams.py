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
from threading import Thread, Event

class MicrophoneStream:
	def __init__(self, detector):
		self.chunk_size = 1600
		self.sample_format = pyaudio.paInt16 
		self.channels = 1
		self.frame_rate = 16000
		self.seconds = 3

		self.detector = detector
	
	def start(self):
		p = pyaudio.PyAudio() 
		self.stream = p.open(format=self.sample_format,
				channels=self.channels,
				rate=self.frame_rate,
				frames_per_buffer=self.chunk_size,
				input=True)

		self.running = True
		self.thread = Thread(target=self._handle_new_audio_chunk, daemon=True)
		self.thread.daemon = True
		self.thread.start()

		self.new_chunk_event = Event()

		num_of_required_frames = self.frame_rate * self.seconds/ self.chunk_size
		return num_of_required_frames
	
	def stop(self):
		if self.thread:
			self.running = False
			self.thread.join()
			self.thread = None

	def _handle_new_audio_chunk(self):
		while self.running:
			chunk = self.stream.read(self.chunk_size)
			self.detector.add_audio_chunk(chunk)
			self.new_chunk_event.set()
