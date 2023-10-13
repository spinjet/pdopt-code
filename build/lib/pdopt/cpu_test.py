# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:50:34 2021

@author: s345001
"""

import psutil

print(psutil.cpu_count(logical = False))
print(psutil.cpu_count())