#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
import time

def timing_val(func):
	def wrapper(*arg, **kw):
		t1 = time.time()
		res = func(*arg, **kw)
		t2 = time.time()
		print((t2 - t1), res, func.__name__)
	return wrapper
