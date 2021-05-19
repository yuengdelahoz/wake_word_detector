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
	utils.generate_background_noise_clips("/audio_files/background_noise","/audio_files/background_noise_chunks")
	prepare_hey_ida_data("/audio_files/originals/")
	prepare_mozilla_common_data("/audio_files2")

def create_dataset():
	from data import generate_dataset
	generate_dataset()

def audiosegment_to_librosawav(audiosegment):    
	channel_sounds = audiosegment.split_to_mono()
	samples = [s.get_array_of_samples() for s in channel_sounds]

	fp_arr = np.array(samples).T.astype(np.float32)
	fp_arr /= np.iinfo(samples[0].typecode).max
	fp_arr = fp_arr.reshape(-1)

	return fp_arr

if __name__ == "__main__":
	# prepare_data()
	create_dataset()
	


