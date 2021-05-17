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
from . import utils
import numpy as np
from pydub import AudioSegment, silence, effects
import math
import traceback
import sys
from glob import glob

BACKGROUND_NOISE_FOLDER = None
start, end = 0, 3
def get_next_background_noise_chunk(audio_file):
	global start, end
	chunk = BACKGROUND_NOISE_FOLDER[start:end]
	n_chunks = list()
	for n_chunk_path in chunk:
		try:
			noise = AudioSegment.from_file(n_chunk_path) 
			noise.apply_gain(-noise.dBFS*0.5)
			# noise= AudioSegment.silent(3000) 
			# noise = noise.apply_gain(audio_file.dBFS)
			n_chunks.append(noise)
		except:
			traceback.print_exc()
	start += 3
	end += 3
	if start >= len(BACKGROUND_NOISE_FOLDER) or end >= len(BACKGROUND_NOISE_FOLDER):
		start = 0
		end = 0
	return n_chunks


def speed_change(sound, speed=1.0):
	# Manually override the frame_rate. This tells the computer how many
	# samples to play per second
	sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
		 "frame_rate": int(sound.frame_rate * speed)
	  })
	 # convert the sound with altered frame rate to a standard frame rate
	 # so that regular playback programs will work right. They often only
	 # know how to play audio at standard frame rate (like 44.1k)
	return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def _speed_update(audio_file):
	clips = list()
	speeds = [0.9,1.2]
	for sp in speeds:
		new_audio = speed_change(audio_file,sp)
		rate = sp
		while new_audio.duration_seconds > 3:
			rate = rate + 0.01
			new_audio = speed_change(audio_file,rate)
		clips.append(new_audio)
	clips.append(audio_file)
	return clips

def _gain_update(audio_file):
	clips = list()
	gains = [-5,10]
	for g in gains:
		new_audio = audio_file.apply_gain(g)
		clips.append(new_audio)
	clips.append(audio_file)
	return clips

def _add_background_noise(audio_file,duration_thres):
	clips = list()
	noise_chunks = get_next_background_noise_chunk(audio_file)
	end = duration_thres - len(audio_file)
	mid = int(end/2)
	positions = [0,mid,end]
	# positioning the main audio clip around the background noise
	for i,pos in enumerate(positions):
		if pos < 0:
			continue
		n_chunk = noise_chunks[i]
		new_audio = n_chunk.overlay(audio_file,position=pos,gain_during_overlay=6)
		clips.append(new_audio)

	# positioning the main audio clip around the silence noise
	silence = AudioSegment.silent(duration_thres) 
	new_audio =silence.overlay(audio_file,position=mid)
	clips.append(audio_file)
	return clips


def _augment_in_time_domain(audio_file, duration_thres):
	augmented_clips = list()

	s_clips = _speed_update(audio_file)
	# print('s_clips',len(s_clips))
	g_clips = 0
	g_clips_cnt =0
	n_clips_cnt =0
	for sclip in s_clips:
		g_clips = _gain_update(sclip)
		# print('g_clips',len(g_clips))
		for gclip in g_clips:
			g_clips_cnt +=1
			n_clips = _add_background_noise(gclip,duration_thres)
			# print('n_clips',len(n_clips))
			for nclip in n_clips:
				augmented_clips.append(nclip)
	# print('total g_clips',g_clips_cnt)
	# print('total n_clips',len(augmented_clips))
	return augmented_clips

def _augment_audio(audio_file, duration_thres):
	return _augment_in_time_domain(audio_file, duration_thres)

def _fix_duration_and_convert_audio(AUDIO_FILE_PATH, OUTPUT_FOLDER, duration_thres = 3000, file_size_thresh = 100, min_highest_amplitude=-25, augment = False):
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
							print(cnt,file_path)
							clips = list()
							if augment:
								clips = _augment_audio(audio_file,duration_thres)
							else:
								start = int((duration_thres - duration)/2)
								new_audio = AudioSegment.silent(duration_thres) 
								new_audio = new_audio.overlay(audio_file, position=start)
								clips.append(new_audio)

							for clip in clips:
								new_audio = clip.set_channels(1)
								new_audio = new_audio.set_sample_width(2)
								new_audio = new_audio.set_frame_rate(16000)
								out_file = os.path.join(OUTPUT_FOLDER,'file_{:04d}.wav'.format(cnt))
								new_audio.export(out_file,format="wav")
								# print(cnt,file_path, "gain: {}".format(clip.dBFS))
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
		audio_clips = audio_clips[:10000]

		#save audio clips 
		for aud_clip in audio_clips:
			src_file = os.path.join(OUTPUT_FOLDER, aud_clip) 
			dst_file = os.path.join(OUTPUT_FOLDER_2, aud_clip) 
			shutil.copyfile(src_file, dst_file)
			print('copying',src_file,dst_file)
	else:
		print("Noting to do")
	

def prepare_hey_ida_data(AUDIO_FILE_PATH):
	OUTPUT_FOLDER = "/audio_files/one_class"
	global BACKGROUND_NOISE_FOLDER
	BACKGROUND_NOISE_FOLDER = glob("/audio_files/background_noise_chunks/*.wav")
	np.random.shuffle(BACKGROUND_NOISE_FOLDER)
	_fix_duration_and_convert_audio(AUDIO_FILE_PATH, OUTPUT_FOLDER,file_size_thresh = 1000, min_highest_amplitude = -1000, augment = True)
