from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Author : Mathis Antonetti
# Date : 18/03/2022
# Description : This is a simulation code to compute the resistor of linearised 1D Saint-Venant equations with space-varying slope and uniform friction using a finite difference scheme (explicit Euler order 1).

### Settings & Parameters

# Physics
g = 9.81 # gravity acceleration
L = 8000 # length
k = 0.002 # friction

# Initial conditions

Het0 = 2 # stationary height H1*(0) at the start
Vet0 = 0.5 # stationary speed V1*(0) at the start

# Settings of simulation parameters
dx = 1 # space step

# Computation of other important parameters
nbstepx = int(L / dx)
Lx = np.linspace(0, L, nbstepx)

# Bathymetry (slope)

C = np.zeros(nbstepx)

### Saint-Venant's linearized PDE fonctions

def lambda1(Vet, Het):
    return Vet + np.sqrt(g*Het)

def lambda2(Vet, Het):
    return np.sqrt(g*Het) - Vet

def delta1(f, Vet, Het, bathy):
    return (f * (Vet**2) / Het) * (-1 / (4 * lambda1(Vet, Het)) + 1 / Vet + 1 / (2 * np.sqrt(g * Het))) + bathy / (4 * lambda1(Vet, Het))

def delta2(f, Vet, Het, bathy):
    return (f * (Vet**2) / Het) * (3 / (4 * lambda2(Vet, Het)) + 1 / Vet + 1 / (2 * np.sqrt(g * Het))) - 3*bathy / (4 * lambda2(Vet, Het))

def gamma1(f, Vet, Het, bathy):
    return (f * (Vet**2) / Het) * (-3 / (4 * lambda1(Vet, Het)) + 1 / Vet - 1 / (2 * np.sqrt(g * Het))) + 3*bathy / (4 * lambda1(Vet, Het))

def gamma2(f, Vet, Het, bathy):
    return (f * (Vet**2) / Het) * (1 / (4 * lambda2(Vet, Het)) + 1 / Vet - 1 / (2 * np.sqrt(g * Het))) - bathy / (4 * lambda2(Vet, Het))

### Algorithm & Computation

# Computes the stationnary states
def stationary_state(x0):
    x = np.zeros((nbstepx, 2)) # (H*, V*)
    x[0] = x0
    for j in range(nbstepx - 1):
        x[j+1, 1] = x[j, 1] + dx * (k*x[j, 1]**2/x[j, 0] - C[j])*x[j, 1]/(g*x[j, 0] - x[j, 1]**2)
        x[j+1, 0] = x[j, 0] - (x[j+1, 1] - x[j, 1])*x[j, 0]/x[j, 1]
    return x

# Initialization of the stationary states
Hp = np.zeros(nbstepx)
Vp = np.zeros(nbstepx)
Zf = np.zeros(nbstepx)

# Computation of the stationnary states
stationary = stationary_state(np.array([Het0, Vet0]))
Hp, Vp = stationary[:, 0], stationary[:, 1]
Zf = np.array([dx*np.sum(C[:i]) for i in range(nbstepx)])

print("Simulation of the stationary states : success")

# Computes the resistor function
def resistor(eta0, Hstar, Vstar):
    phi = 1.0 # (phi(0) = 1)
    resistor = np.zeros(nbstepx) + eta0
    for i in range(nbstepx - 1):
        phi = phi*np.exp(dx*(gamma1(k, Vstar[i], Hstar[i], C[i])/lambda1(Vstar[i], Hstar[i]) + delta2(k, Vstar[i], Hstar[i], C[i])/lambda2(Vstar[i], Hstar[i])))
        resistor[i+1] = resistor[i] + dx*(phi*delta1(k, Vstar[i], Hstar[i], C[i])/lambda1(Vstar[i], Hstar[i]) + gamma2(k, Vstar[i], Hstar[i], C[i])/(lambda2(Vstar[i], Hstar[i])*phi)*(resistor[i]**2))

    return resistor

res = resistor(1.5, Hp, Vp)

# Computes the maximal y0 such that the resistor eta is defined with eta(0) = y0, assuming C = 0
def max_init_cond(r0):
    return (1-r0)/(1+r0)+np.power((1+r0)/(4*r0*(1-r0))*(2*r0**2 - 2*r0 + 1 - np.power(r0, 8/3)), -1)

r_0 = np.linspace(0.001, 0.999, 1000)
y0_star = max_init_cond(r_0)

### Display
# useful functions
def contours(L):
    xmin = Lx[0] - 5
    xmax = Lx[nbstepx - 1] + 5
    hmin = np.min(L)
    hmax = np.max(L)
    espace = (hmax - hmin) / 20
    hmin -= espace
    hmax += espace
    return xmin, xmax, hmin, hmax

def Display_GP(val_name, unit, Val):
    xmin, xmax, hmin, hmax = contours(Val)
    plt.xlim = (xmin, xmax)
    plt.ylim = (hmin, hmax)
    plt.plot(Lx, Val)
    plt.title(val_name + "* (" + unit + ") varying with x")

    plt.show()

# stationary state
Display_GP('H', 'm', Hp - Zf) # Height
Display_GP('V', 'm/s', Vp) # Speed
Display_GP('eta', 'N.A.', res) # Resistor

# Globality Condition
plt.title("y0_max varying with r(0)")
plt.plot(r_0, y0_star)
plt.show()