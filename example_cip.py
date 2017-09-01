#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
example_cip.py
--------------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from cip import Sod
import numpy as np

# Users can run the Sod example...
solver = Sod(CSL2 = False)
solver.Animate(thin = 10)