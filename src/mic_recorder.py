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
# import wave
from pydub import AudioSegment
import numpy as np
from python_speech_features import mfcc
import time
import scipy.io.wavfile as wav

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


	
chunk = 1600  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second
seconds = 1
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
		channels=channels,
		rate=fs,
		frames_per_buffer=chunk,
		input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
chunks_in_time = fs * seconds / chunk
print(chunks_in_time)
for i in range(0, int(chunks_in_time)):
	data = read_chunk(stream,chunk)
	frames.append(data)


# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

bytestring =b''.join(frames)
audio_clip = np.frombuffer(bytestring, dtype=np.int16)
path = 'output.wav'

# wav.write(path,fs, audio_clip)
# mfcc_psf1=  _generate_mfcc(audio_clip ,fs)
# print('mfcc_psf1',mfcc_psf1)

# (rate,sig) = wav.read(path)
# mfcc_psf2 = mfcc(sig,rate)
# print('mfcc_psf2',mfcc_psf2)

# # print('mfcc_psf1 == mfcc_psf2', np.equal(mfcc_psf1,mfcc_psf2))
# print('mfcc_psf1 == mfcc_psf2', np.array_equal(mfcc_psf1,mfcc_psf2))

