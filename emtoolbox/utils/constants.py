#!/usr/bin/python3

'''Constants for use in EMToolBox'''

import math

VP0 = 299792458
MU0 = 4e-7 * math.pi
EPS0 = 1 / (VP0**2 * MU0)
ETA0 = MU0 * VP0

# Average, low-frequency at room temperature
COND_AG = 6.17e7
COND_CU = 5.80e7
COND_AU = 4.10e7
COND_AL = 3.54e7

# Colors
RGB_CU = '#b87333'
RGB_AL = '#848789'
RGB_DIEL = '#a5a5a5'

# Special characters
CHR_OHM = chr(0x3a9)
