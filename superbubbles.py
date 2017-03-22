#!/usr/bin/python3

from numpy import pi, sqrt, arccos, exp
import scipy.integrate as integrate
import numpy as np
import plawt

# Scaling
H = 1 # [L] = H
gamma = 5/3
L_not = 1

def r(z,y):
	arg = 1 - y**2/(4*H**2) + exp(-z/H)
	arg *= exp(z/(2*H))/2
	return 2 * H * arccos(arg)

z = np.arange(-2, 10, 0.00001)
y = [0.5, 1, 1.4, 1.7, 1.9, 1.98, 2.0]
figure1 = {
	'ylabel': 'z/H', 'xlabel': 'r/H',
	'filename': 'shockfront.png',
	'ylim': (-2, 10), 'xlim': (-6, 6),
	'figsize': (6/1.3, 6.5/1.3)
	# 'show': True
}

for i, yi in enumerate(y):
	figure1[i] = {'x': np.concatenate((r(z, y[i]), -r(z, y[i]))), 'y': np.concatenate((z,z))}
plawt.plot(figure1)

# Omega = pi * scipy.integrate(r(z, t)**2, z, z2, z1) # (A3)
# P = (gamma - 1) * thermalE / Omega # (A2)

# dthermalEdt = L_not - P*dOmegadt
# dydt = sqrt(((gamma**2 - 1) * thermalE)/(2*(rho_not * Omega))) # (A6)
