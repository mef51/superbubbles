#!/usr/bin/python3

from numpy import pi, sqrt, arccos, arcsin, exp, log
from tqdm import tqdm
import scipy.integrate as integrate
import numpy as np; np.seterr(invalid='ignore')
import plawt

def r(z,y):
	""" Get the shape of the shockfront """
	arg = 1 - y**2/(4*H**2) + exp(-z/H)
	arg *= exp(z/(2*H))/2
	return 2 * H * arccos(arg)

def z12(y):
	"""
	Get the edges of the shockfront
	returns tuple (z1, z2)
	"""
	return (-2*H*log(1 - y/(2*H)), -2*H*log(1 + y/(2*H)))

def rmax(y):
	""" Get max radius of the bubble """
	return 2*H*arcsin(y/(2*H))


def shockfronts():
	z = np.arange(-2, 10, 0.00001)
	y = [0.1, 0.5, 1, 1.4, 1.7, 1.9, 1.98, 2.0]
	figure1 = {
		'ylabel': 'z/H', 'xlabel': 'r/H',
		'filename': 'shockfront.png',
		'ylim': (-2, 10), 'xlim': (-6, 6),
		'figsize': (6/1.3, 6.5/1.3),
		'show': False
	}

	for i, yi in enumerate(tqdm(y)):
		figure1[i] = {'x': np.concatenate((r(z, y[i]), -r(z, y[i]))), 'y': np.concatenate((z,z))}
	plawt.plot(figure1)

# Scaling
H = 1 # [L] = H
gamma = 5/3
L_not = 1
rho_not = 1
P = 1

# initial conditions
yi = 0.01 #(goes up to 1.98?)
Omegai = (4*pi/3)*rmax(yi)**3
Ethi = P/(gamma-1)*Omegai

# use quad to find dOmega, but use odeint to solve the entire system

dy = lambda Eth, Omega: sqrt((gamma**2 - 1)*Eth / 2  / (rho_not * Omega))
dr = lambda z, y: y / ( 2*sqrt(1 - 1/4*exp(z/H)*(1-y**2/(4*H**2)+exp(-z/H))**2) )
dOmega = lambda y: 2 * pi * integrate.quad(lambda z: r(z, y) * dr(z, y), z12(y)[1], z12(y)[0])[0]
dEth = lambda y: L_not - P * dOmega(y)

def bubblesystem(state, t):
	y, Omega, Eth = state

	# need dydt, drdt, dOmegadt, dEthdt
	dydt = dy(Eth, Omega)
	dOmegadt = dOmega(y)
	dEthdt = L_not - P * dOmegadt

	return [dydt, dOmegadt, dEthdt]

initialstate = [yi, Omegai, Ethi]
time = np.arange(0.2, 1, 0.001)
results = integrate.odeint(bubblesystem, initialstate, time)
print(len(results))
