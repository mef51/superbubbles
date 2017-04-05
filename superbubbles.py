#!/usr/bin/python3

from numpy import pi, sqrt, arccos, arcsin, exp, log
from tqdm import tqdm
import scipy.integrate as integrate
import numpy as np
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

# Dimensionless Scaling
H = 1 # [L] = H
gamma = 5/3
L_not = 1
rho_not = 1
P = 1

# Math Helpers
dy = lambda Eth, Omega: sqrt((gamma**2 - 1)*Eth / 2  / (rho_not * Omega))
dr = lambda z, y: y / ( 2*sqrt(1 - 1/4*exp(z/H)*(1-y**2/(4*H**2)+exp(-z/H))**2) )
dOmega = lambda y: 2 * pi * integrate.quad(lambda z: r(z, y) * dr(z, y), z12(y)[1], z12(y)[0])[0]
dEth = lambda y: L_not - P * dOmega(y)

OmegaFunc = lambda y: pi * integrate.quad(lambda z: r(z, y)**2, z12(y)[1], z12(y)[0])[0]
EnergyFunc = lambda oprev, onext, e: oprev**(gamma-1)*(e+dt)/(onext**(gamma-1))

# initial conditions
dt = 0.0001
time = np.arange(0.005, 10, dt)
yi = 0.01
Omegai = OmegaFunc(yi)
Ethi = P/(gamma-1)*Omegai

initialstate = [yi, Omegai, Ethi]
ys = [yi]
Omegas = [Omegai]
Es = [Ethi]

# Integrate
for t in tqdm(time):
	ynext = ys[-1] + dy(Es[-1], Omegas[-1])*dt
	omeganext = OmegaFunc(ynext)
	energynext = EnergyFunc(Omegas[-1], omeganext, Es[-1])

	ys.append(ynext)
	Omegas.append(omeganext)
	Es.append(energynext)
	if ynext > 1.99999:
		break

plawt.plot({
	0: {'x': time[:len(ys)-1], 'y': ys[:-1]},
    'show':False,
    'filename': 'yplot.png',
    'title': "Y",
    'xlabel': 'time',
	'set_yscale': 'log', 'set_xscale': 'log',
	'xlim': (0.01, 10), 'ylim': (0.1, 10.0)
})
plawt.plot({
	0: {'x': time[:len(ys)-1], 'y': Omegas[:-1]},
    'show':False,
    'filename': 'Omegaplot.png',
    'title': "Omega",
    'xlabel': 'time',
	'set_yscale': 'log', 'set_xscale': 'log',
	'xlim': (0.01, 10), 'ylim': (0.1, 10.0)
})
plawt.plot({
	0: {'x': time[:len(ys)-1], 'y': Es[:-1]},
    'show':False,
    'filename': 'Energyplot.png',
    'title': "Energy",
    'xlabel': 'time',
	'set_yscale': 'log', 'set_xscale': 'log',
	'xlim': (0.01, 10), 'ylim': (0.01, 10.0)
})


# alternative: use odeint to solve the entire system. doesn't work
def bubblesystem(state, t):
	y, Omega, Eth = state

	# need dydt, drdt, dOmegadt, dEthdt
	dydt = dy(Eth, Omega)
	dOmegadt = dOmega(y)
	dEthdt = L_not - P * dOmegadt

	return [dydt, dOmegadt, dEthdt]

# results = integrate.odeint(bubblesystem, initialstate, time)
# print(time[-1])
# plawt.plot({0:{'x':time, 'y':results[:, 0]}, 'show':False, 'filename': 'yplot2.png',     'title': 'y'})
# plawt.plot({0:{'x':time, 'y':results[:, 1]}, 'show':False, 'filename': 'Omegaplot2.png', 'title': 'Omega'})
# plawt.plot({0:{'x':time, 'y':results[:, 2]}, 'show':False, 'filename': 'Energyplot2.png', 'title': 'Energy'})

