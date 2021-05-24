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
import shutil
import utils
import numpy as np
from pydub import AudioSegment, silence, effects
import math
import traceback
from glob import glob

def _split_originals(AUDIO_FILE_PATH):
	output_folder = "/audio_files/dataset/original_splits"
	if os.path.exists(output_folder):
		return output_folder
	audio_files = list()
	for root, dirnames, filenames in os.walk(AUDIO_FILE_PATH):
		for f in filenames:
			ext = f.split('.')[-1]
			if ext in ['mp3','wav','m4a']:
				file_path = os.path.join(root,f)
				audio_files.append(file_path)
	np.random.shuffle(audio_files)
	train_cnt = int(np.ceil(len(audio_files)*0.8))

	for src_path in audio_files:
		folder_name=''
		if train_cnt > 0:
			train_cnt -= 1
			folder_name='train'
		else:
			folder_name='test'
			
		dirname = os.path.basename(os.path.dirname(src_path))
		outname = os.path.basename(src_path)
		dst_path = os.path.join(output_folder,folder_name,dirname)
		utils.create_folder(dst_path)
		dst_path = os.path.join(dst_path,outname)
		shutil.copyfile(src_path,dst_path)
		print(src_path,dst_path)
	return output_folder


BACKGROUND_NOISE_FOLDER = None
n_idx = 0
def get_next_background_noise_chunk(audio_file):
	global n_idx
	chunk = BACKGROUND_NOISE_FOLDER[n_idx:n_idx+3]
	n_chunks = list()
	for n_chunk_path in chunk:
		try:
			noise = AudioSegment.from_file(n_chunk_path) 
			noise = noise.apply_gain(noise.dBFS*0.2)
			n_chunks.append(noise)
		except:
			traceback.print_exc()
	n_idx += 3
	if n_idx > len(BACKGROUND_NOISE_FOLDER)-3:
		np.random.shuffle(BACKGROUND_NOISE_FOLDER)
		n_idx = 0
	assert len(n_chunks) == 3
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
	if audio_file.duration_seconds < 1.5:
		speeds = [0.75,0.85]
	else:
		speeds = [0.9,1.1]

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
	if len(noise_chunks) != 3:
		input('differente chunks')
	end = duration_thres - len(audio_file)
	mid = int(end/2)
	positions = [0,mid,end]
	# positioning the main audio clip around the background noise
	for i,pos in enumerate(positions):
		if pos < 0:
			continue
		n_chunk = noise_chunks[i]
		new_audio = n_chunk.overlay(audio_file,position=pos,gain_during_overlay=10)
		clips.append(new_audio)

	# positioning the main audio clip around the silence noise
	silence = AudioSegment.silent(duration_thres) 
	new_audio =silence.overlay(audio_file,position=mid)
	clips.append(new_audio)
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
		for gclip in g_clips:
			g_clips_cnt +=1
			n_clips = _add_background_noise(gclip,duration_thres)
			for nclip in n_clips:
				augmented_clips.append(nclip)
	return augmented_clips

def _augment_audio(audio_file, duration_thres):
	return _augment_in_time_domain(audio_file, duration_thres)

def _fix_duration_and_convert_audio(AUDIO_FILE_PATH, OUTPUT_FOLDER, duration_thres = 3000, file_size_thresh = 100, min_highest_amplitude=-25, augment = False):
	utils.create_folder(OUTPUT_FOLDER)
	exit = False
	global_cnt = 0

	for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
		try:
			if root == OUTPUT_FOLDER:
				continue
			for d in dirs:
				if 'MACOSX' in d:
					print ('deleting {}/{}'.format(root,d))
					shutil.rmtree(os.path.join(root,d))
			
			if augment:
				pattern = os.path.join(OUTPUT_FOLDER,'{}_file_*.wav'.format(os.path.basename(root)))
				already_converted = glob(pattern)
				msg = ""
				if len(already_converted) > 0 and len(filenames)*36 == len(already_converted): # 36 new audio clips are created per original audio clip
					print(root,"-> DONE")
					continue

			cnt = 0
			if 'test' in root:
				OUTPUT_FOLDER2 = os.path.join(OUTPUT_FOLDER,'test')
			elif 'train' in root:
				OUTPUT_FOLDER2 = os.path.join(OUTPUT_FOLDER,'train')

			utils.create_folder(OUTPUT_FOLDER2)

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
							global_cnt +=1
							print(global_cnt,file_path,"-->")
							clips = list()
							if augment and 'test' not in root:
								clips = _augment_audio(audio_file,duration_thres)
							else:
								start = int((duration_thres - duration)/2)
								new_audio = AudioSegment.silent(duration_thres) 
								new_audio = new_audio.overlay(audio_file, position=start)
								clips.append(new_audio)

							for clip in clips:
								out_file = os.path.join(OUTPUT_FOLDER2,'{}_file_{:08d}.wav'.format(os.path.basename(root),cnt))
								new_audio = clip.set_channels(1)
								new_audio = new_audio.set_sample_width(2)
								new_audio = new_audio.set_frame_rate(16000)
								new_audio.export(out_file,format="wav")
								cnt += 1
								print(global_cnt,file_path,"-->", out_file)
								if len(new_audio) != 3000:
									input("{} wrong duration, {} ".format(file_path, len(new_audio)))

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
	OUTPUT_FOLDER_2 ="/audio_files/dataset/classes/non-target"

	exists = os.path.exists(OUTPUT_FOLDER_2)
	if not exists  or ( exists and len(os.listdir(OUTPUT_FOLDER_2)) != 10000):
		utils.create_folder(OUTPUT_FOLDER_2 )

		#Choose 10000 audio clips randomly
		# np.random.shuffle(audio_clips)
		# audio_clips = audio_clips[:10000]

		#save audio clips 
		for aud_clip in audio_clips:
			src_file = os.path.join(OUTPUT_FOLDER, aud_clip) 
			dst_file = os.path.join(OUTPUT_FOLDER_2, aud_clip) 
			shutil.copyfile(src_file, dst_file)
			print('copying',src_file,dst_file)
	else:
		print("Noting to do")
	
def prepare_hey_ida_data(AUDIO_FILE_PATH):
	AUDIO_FILE_PATH = _split_originals(AUDIO_FILE_PATH)
	OUTPUT_FOLDER = "/audio_files/dataset/classes/target"

	global BACKGROUND_NOISE_FOLDER
	BACKGROUND_NOISE_FOLDER = glob("/audio_files/background_noise_chunks/*.wav")
	np.random.shuffle(BACKGROUND_NOISE_FOLDER)
	_fix_duration_and_convert_audio(AUDIO_FILE_PATH, OUTPUT_FOLDER,file_size_thresh = 1000, min_highest_amplitude = -1000, augment = True)
