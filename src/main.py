#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""

def prepare_data():
	from data import prepare_mozilla_common_data, prepare_hey_ida_data
	from data import utils 
	# utils.generate_background_noise_clips("/audio_files/background_noise","/audio_files/background_noise_chunks")
	prepare_hey_ida_data("/audio_files/originals/")
	# prepare_mozilla_common_data("/audio_files2")

def create_dataset():
	from data import generate_dataset
	generate_dataset()

def train_and_export():
	from train import build_and_train
	build_and_train()

def analyze_mic_data():
	from detection import Detector
	detector = Detector()
	def callback(pred):
		# idx,conf = pred
		if pred > 0.6:
			print(pred,'what up')
	
	detector.start(callback)


if __name__ == "__main__":
	# prepare_data()
	# create_dataset()
	# train_and_export()
	analyze_mic_data()
	


