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
		t1 = time.time()*1000
		res = func(*arg, **kw)
		t2 = time.time()*1000
		print('{} took {:.2f} ms'.format(func.__name__,(t2 - t1)))
		return res
	return wrapper
