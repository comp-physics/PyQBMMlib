#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:06:28 2021

@author: alexis
"""

import numpy as np

class loss_params:
    def __init__(self, Re, Pressure,ids,npoints, mom_scale_coeffs, rhs_scale_coeffs):
        self.Re = Re
        self.Pressure = Pressure
        self.ids = ids
        self.npoints = npoints
        self.mom_scale_coeffs = mom_scale_coeffs
        self.rhs_scale_coeffs = rhs_scale_coeffs