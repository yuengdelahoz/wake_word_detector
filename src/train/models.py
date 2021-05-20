#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU

def build_model(input_shape):
	model = Sequential([
		LSTM(128,input_shape=input_shape),
		Dense(64, activation='relu'),
		Dropout(0.3),
		Dense(2, activation='softmax')
		])
	return model

