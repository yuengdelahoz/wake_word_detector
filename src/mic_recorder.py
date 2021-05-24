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
import wave
from pydub import AudioSegment
import numpy as np
from python_speech_features import mfcc
import time
from time import sleep
import os

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
output_folder = os.path.join(dir_path,os.pardir,"audio_files","microphone_audio_files")

def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

def timing_val(func):
	def wrapper(*arg, **kw):
		t1 = time.time()*1000
		f = func(*arg, **kw)
		t2 = time.time()*1000
		print('It took {:.2f} ms to execute function {}'.format((t2 - t1), func.__name__))
		return f
	return wrapper

@timing_val
def _generate_mfcc(audio_clip,fs):
	return mfcc(audio_clip, fs)

@timing_val
def read_chunk(stream,chunk):
	return stream.read(chunk)

create_folder(output_folder)
	
chunk_size = 1600  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second
seconds = 3

p = pyaudio.PyAudio()  # Create an interface to PortAudio

frames = list()

chunks_in_time_window = fs * seconds / chunk_size
cnt = 0
def callback(in_data, frame_count, time_info, flag):
	global frames
	global cnt
	frames.append(in_data)
	if len(frames) > chunks_in_time_window:
		while len(frames) != chunks_in_time_window:
			frames.pop(0)

	if len(frames) == chunks_in_time_window:
		bytestring =b''.join(frames)
		audio_clip = np.frombuffer(bytestring, dtype=np.int16)
		path = os.path.join(output_folder,'voice_{:08d}.wav'.format(cnt))
		wf = wave.open(path , 'wb')
		wf.setnchannels(channels)
		wf.setsampwidth(p.get_sample_size(sample_format))
		wf.setframerate(fs)
		wf.writeframes(bytestring)
		wf.close()
		cnt +=1

	return None, pyaudio.paContinue


stream = p.open(format=sample_format,
		channels=channels,
		rate=fs,
		frames_per_buffer=chunk_size,
		input=True,
		output=False,
		stream_callback= callback
		)

print('Recording')
t = 6
while t > 0:
	try:
		if t % 3 == 0:
			print('go')
		sleep(1)
		t -= 1

	except KeyboardInterrupt:
		break
	


# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()
print('Finished recording')
