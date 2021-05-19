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
from python_speech_features import mfcc
import numpy as np
import glob
import traceback
import json
from collections import namedtuple
import librosa

def _generate_mfcc(audio_file):
	audio_clip = np.frombuffer(audio_file.get_array_of_samples(), dtype=np.int16)
	return mfcc(audio_clip, samplerate=audio_file.frame_rate)

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
					if global_cnt % 100 == 0:
						print(global_cnt,'processing',file_path,'mfcc',mfcc_feat.shape,'label',label_id)
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

def _split_dataset2(dataset):
	labels = dataset['labels']
	mfccs = dataset['mfcc']
	paths = dataset['audio_path']

	split_names = ['train','test','validation']
	Splits = namedtuple('splits',split_names)
	splits = Splits(0.8,0.15,0.05)

	split_dataset = {}
	indices = list(range(0,len(labels)))
	np.random.shuffle(indices)
	total = 0
	for split_name in split_names:
		split_value = splits._asdict()[split_name]
		if split_name not in split_dataset:
			if split_name == 'validation':
				num_of_elements = len(indices) - total
			else:
				num_of_elements =  int(np.ceil(len(indices) * split_value))
			total = total + num_of_elements
			split_dataset[split_name] = {'labels':list(),'mfccs':list(), 'counter':num_of_elements }

	idx = 0
	for i in indices:
		label = labels[i]
		mfcc = mfccs[i]
		path = paths[i]
		was_added = False
		loop_counter = 3
		while loop_counter > 0 and not was_added:
			split_name = split_names[idx]
			counter = split_dataset[split_name]['counter']
			if counter > 0:
				counter -= 1
				split_dataset[split_name]['labels'].append(label)
				split_dataset[split_name]['mfccs'].append(mfcc)
				split_dataset[split_name]['counter'] = counter
				was_added = True
				print(path,'was added')
			else:
				idx = (idx +1) % len(split_names)
			loop_counter -= 1

		if not was_added: # nothing do 
			print('nothing was added')
			break
	
	counter = 0
	for split_name,data in split_dataset.items():
		print(split_name,len(data['labels']),len(data['mfccs']))
		counter += len(data['labels'])
		data.pop('counter')
	print('total',counter)
	assert counter == len(labels)
	return split_dataset

def _split_dataset(dataset):
	# k = 11
	# dataset = {
			# 'labels':np.random.randint(0,2,size=(k)).tolist(),
			# 'mfcc':np.random.random(size=(k)).tolist()
			# }
	labels = dataset['labels']
	mfccs = dataset['mfcc']

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


def generate_dataset():
	AUDIO_FILE_PATH = "/audio_files/dataset/classes"
	OUTPUT_FOLDER = "/audio_files/dataset"
	dataset =_generate_dataset(AUDIO_FILE_PATH,OUTPUT_FOLDER)
	# dataset = []
	split_dataset= _split_dataset(dataset)
	out_file = os.path.join(OUTPUT_FOLDER,"dataset_split.json")
	with open(out_file, "w") as fp:
		json.dump(split_dataset,fp, indent=6)

























