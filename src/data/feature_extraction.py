#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
from . import utils
from pydub import AudioSegment
import os

import pydub
import numpy as np
from glob import glob
import traceback
import json
from collections import namedtuple
import librosa
from utils import timing_val
from python_speech_features import mfcc
import librosa

@timing_val
def _generate_dataset(AUDIO_FILE_PATH, OUTPUT_FOLDER):
	out_file = os.path.join(OUTPUT_FOLDER,"dataset.json")
	if os.path.exists(out_file):
		print('loading dataset')
		with open(out_file, "r") as fp:
			dataset = json.load(fp)
	else:
		utils.create_folder(OUTPUT_FOLDER)
		exit = False
		global_cnt = 0
		dataset = {
				"mappings":{},
				"audio_path":[],
				"labels":[],
				"mfcc":[]
				}
		for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
			if root == AUDIO_FILE_PATH:
				continue

			label_name = os.path.basename(root)
			label_id = 0
			if label_name == "one_class":
				dataset['mappings'].update({"target": 1})
				label_id = 1
			elif label_name == "zero_class":
				dataset['mappings'].update({"non-target": 0})

			for f in filenames:
				file_path = os.path.join(root,f)
				try:
					audio_file = AudioSegment.from_file(file_path)
					mfcc_feat = _generate_mfcc(audio_file)
					dataset['mfcc'].append(mfcc_feat.tolist())
					dataset['labels'].append(label_id)
					dataset['audio_path'].append(file_path)
					global_cnt += 1
					if global_cnt % 50 == 0:
						print(global_cnt,'processing',file_path,'mfcc',mfcc_feat.shape,'label',label_id)
						break
				except KeyboardInterrupt:
					exit = True
					break
				except:
					traceback.print_exc()

				if exit:
					break
		with open(out_file, "w") as fp:
			json.dump(dataset,fp, indent=6)
	print('dataset loaded')
	return dataset

def _split_dataset(dataset):
	labels = dataset['labels']
	mfccs = dataset['mfccs']

	data = {}
	for i in range(len(labels)):
		label_id = labels[i]
		if label_id not in data:
			data[label_id] = list()
		data[label_id].append(i)

	split_names = ['train','test','validation']
	Splits = namedtuple('splits', split_names)
	splits = Splits(0.8,0.15,0.05)

	dataset_temp = {}
	for label_id,indices in data.items():
		start = 0
		np.random.shuffle(indices)
		for split_name, split_value in splits._asdict().items():
			offset = int(np.ceil(len(indices) * split_value))
			end = start + offset
			split_indices = indices[start:end]

			if split_name == 'validation':
				split_indices = indices[start:]
			
			print('label id',label_id,split_name,'offset {}, (start,end): {}, indices: {}'.format(offset,(start,end), len(split_indices)))
			if split_name not in dataset_temp:
				dataset_temp[split_name] = {'labels':list(),'mfccs':list()}

			dataset_temp[split_name]['labels'] = dataset_temp[split_name]['labels'] + np.array(labels)[split_indices].tolist()
			dataset_temp[split_name]['mfccs'] =dataset_temp[split_name]['mfccs'] + np.array(mfccs)[split_indices].tolist()
			start = end

	split_dataset = {}
	total = 0
	for split_name, data in dataset_temp.items():
		indices = list(range(0,len(data['labels'])))
		np.random.shuffle(indices)
		if split_name not in split_dataset:
			split_dataset[split_name] = {'labels':list(),'mfccs':list()}
		split_dataset[split_name]['labels'] = np.array(data['labels'])[indices].tolist()
		split_dataset[split_name]['mfccs'] = np.array(data['mfccs'])[indices].tolist()
		total += len(indices)

	print('total',total)
	return split_dataset

@timing_val
def _generate_dataset2(AUDIO_FILE_PATH, OUTPUT_FOLDER):
	out_file = os.path.join(OUTPUT_FOLDER,"dataset.json")
	if os.path.exists(out_file):
		print('loading dataset')
		with open(out_file, "r") as fp:
			dataset = json.load(fp)
	else:
		utils.create_folder(OUTPUT_FOLDER)
		exit = False
		global_cnt = 0
		dataset = {
				"mappings":{},
				"class_counter":{},
				"audio_path":[],
				"labels":[],
				"mfccs":[]
				}

		audio_filenames = list()

		for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
			for f in filenames:
				if f.endswith('.wav'):
					file_path = os.path.join(root,f)
					audio_filenames.append(file_path)

		np.random.shuffle(audio_filenames)

		for file_path in audio_filenames:
			if 'non-target' in file_path:
				label_id = 0
				dataset['mappings'].update({"non-target": 0})
			else:
				label_id = 1
				dataset['mappings'].update({"target": 1})

			if label_id not in dataset['class_counter']:
				dataset['class_counter'][label_id] = 0

			dataset['class_counter'][label_id] += 1

			try:
				audio_file = AudioSegment.from_file(file_path)
				mfcc_feat = _generate_mfcc(audio_file)
				dataset['mfccs'].append(mfcc_feat.tolist())
				dataset['labels'].append(label_id)
				dataset['audio_path'].append(file_path)
				global_cnt += 1
				if global_cnt % 100 == 0:
					print(global_cnt,'processing',file_path,'mfcc',mfcc_feat.shape,'label',label_id)
			except KeyboardInterrupt:
				exit = True
				break
			except:
				traceback.print_exc()

		with open(out_file, "w") as fp:
			json.dump(dataset,fp, indent=6)
	print('dataset generated')
	return dataset

def _split_dataset2(dataset):
	labels = dataset['labels']
	mfccs = dataset['mfccs']
	paths = dataset['audio_path']
	class_counter = dataset['class_counter']

	split_names = ['validation','test','train']
	Splits = namedtuple('splits',split_names)
	splits = Splits(0.05,0.15,0.8)

	split_dataset = {}
	indices = list(range(0,len(labels)))
	np.random.shuffle(indices)
	total = 0
	for split_name in split_names:
		split_value = splits._asdict()[split_name]
		if split_name not in split_dataset:
			counter = dict()
			for label_id, num_of_instances in class_counter.items():
				counter[int(label_id)] = int(np.ceil(num_of_instances * split_value))

			split_dataset[split_name] = {'counter':counter,'paths':list(),'labels':list(), 'mfccs':list()}

	for i in indices:
		label_id = labels[i]
		mfcc = mfccs[i]
		path = paths[i]
		was_added = False
		loop_counter = 3

		idx = 0
		while loop_counter > 0 and not was_added:
			split_name = split_names[idx]
			remain = split_dataset[split_name]['counter'][label_id]
			if remain > 0:
				remain -= 1
				split_dataset[split_name]['labels'].append(label_id)
				split_dataset[split_name]['mfccs'].append(mfcc)
				split_dataset[split_name]['paths'].append(path)
				split_dataset[split_name]['counter'][label_id] = remain
				was_added = True
			else:
				idx = (idx +1) % len(split_names)
			loop_counter -= 1

		if not was_added: # nothing do 
			print('nothing was added')
			break
	
	counter = 0
	for split_name,data in split_dataset.items():
		tmp = dict()
		cnt = 0
		for lid in data['labels']:
			if lid not in tmp:
				tmp[lid] = 0
			tmp[lid] +=1
			cnt +=1
		print(split_name,tmp, 'total',cnt)
		counter += cnt
		split_dataset[split_name]['counter'] = tmp

	print('total',counter)
	assert counter == len(labels)
	return split_dataset

@timing_val
def _generate_dataset3(AUDIO_FILE_PATH, OUTPUT_FOLDER, mfcc_lib="psf", chunk_length = 3000):
	out_file = os.path.join(OUTPUT_FOLDER,"dataset.json")
	sample_rate = 16000
	exit = False
	if os.path.exists(out_file):
		print('loading dataset')
		with open(out_file, "r") as fp:
			dataset = json.load(fp)
	else:
		utils.create_folder(OUTPUT_FOLDER)
		global_cnt = 0
		dataset = {"mfcc_library":mfcc_lib,"mappings":{}}

		audio_filenames = list()
		for root, dirs, filenames in os.walk(AUDIO_FILE_PATH):
			if 'non-target' in root:
				dataset['mappings'].update({"non-target": 0})
			else:
				dataset['mappings'].update({"target": 1})

			for f in filenames:
				if f.endswith('.wav'):
					file_path = os.path.join(root,f)
					audio_filenames.append(file_path)

		np.random.shuffle(audio_filenames)

		for file_path in audio_filenames:
			if 'non-target' in file_path :
				label_id = 0
			else:
				label_id = 1

			split_name = 'train'
			if '/target/' in file_path:
				if 'test' in file_path:
					split_name = 'test'

			if split_name not in dataset:
				dataset[split_name] = {
						"class_counter":{},
						"audio_path":[],
						"labels":[],
						"mfccs":[]
						}

			if label_id not in dataset[split_name]['class_counter']:
				dataset[split_name]['class_counter'][label_id] = 0

			dataset[split_name]['class_counter'][label_id] += 1
			try:
				mfccs = list()
				if mfcc_lib == "librosa":
					audio_file, sr = librosa.load(file_path, sr=sample_rate)
					mfcc_feat = utils.generate_mfcc(audio_file,sample_rate,use='librosa')
				else: # psf
					audio_file = AudioSegment.from_file(file_path)
					mfcc_feat = utils.generate_mfcc(audio_file,sample_rate)

				dataset[split_name]['mfccs'].append(mfcc_feat.tolist())
				dataset[split_name]['labels'].append(label_id)
				dataset[split_name]['audio_path'].append(file_path)
				global_cnt += 1
				if global_cnt % 100 == 0:
					print(global_cnt,'processing',file_path,'mfcc',mfcc_feat.shape,'label',label_id)
					print(duration)
			except KeyboardInterrupt:
				exit = True
				break
			except:
				traceback.print_exc()

	if not exit:
		with open(out_file, "w") as fp:
			json.dump(dataset,fp, indent=6)
	print('dataset generated')
	return dataset

def generate_dataset():
	AUDIO_FILE_PATH = "/audio_files/dataset/classes"
	OUTPUT_FOLDER = "/audio_files/dataset"
	_generate_dataset3(AUDIO_FILE_PATH,OUTPUT_FOLDER)
