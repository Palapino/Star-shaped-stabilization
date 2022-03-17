from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Author : Mathis Antonetti
# Date : 25/02/2022
# Description : This is a simulation code for linearised Saint-Venant equations with space-varying slope and uniform friction using a finite difference scheme (explicit Euler order 1) in the case of a star-shaped net of river branches.

### Settings & Parameters

# Physics
g = 9.81 # gravity acceleration
N = 6 # number of river branches
L = np.full(N, 3000) # length of each branch
L[0] = 3500
L[1] = 3200
k = np.full(N, 0.002) # friction of each branch

# Riemann coordinates control
# Kc = np.full(N, 0.1) # stable
Kc = np.full(N, 20) # unstable

# Initial conditions

Het0 = 2 # stationary height H1*(0) at the start
Vet0 = 1 # stationary speed V1*(0) at the start
weight = np.full(N-2, 1/(N-1)) # weight[i-2] = Vi*(0)/V1*(L_1)
# Condition on weight to respect Kirchoff's laws :
assert(np.sum(weight) < 1)

# Initial perturbations
def v0(x):
    return 0.01 * np.cos(x/100)

def h0(x):
    return 0.1 * np.sin(x/100)

# Settings of simulation parameters
T = 700 # integration time
dx = 10 # space step
CFL = 0.9 # CFL = max(lambda_1, lambda_2)*dt/dx < 1 (Courant-Friedrichs-Lax condition)

# Computation of other important parameters
nbstepx = np.array([int(L[i] / dx) for i in range(N)])
dt = dx*CFL/(np.sqrt(g*Het0) + Vet0)
nbstept = int(T / dt)
nbstepx_max = np.max(nbstepx)

# Bathymetry (slope)

C = [np.zeros(nbstepx[i]) for i in range(N)]

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

# Computes the stationnary states of the branch i
def stationary_state(i, x0):
    x = np.zeros((nbstepx[i], 2)) # (H*, V*)
    x[0] = x0
    for j in range(nbstepx[i] - 1):
        x[j+1, 1] = x[j, 1] + dx * (k[i]*x[j, 1]**2/x[j, 0] - C[i][j])*x[j, 1]/(g*x[j, 0] - x[j, 1]**2)
        x[j+1, 0] = x[j, 0] - (x[j+1, 1] - x[j, 1])*x[j, 0]/x[j, 1]
    return x

# Initialization of the stationary states
Hp = np.zeros((N, nbstepx_max))
Vp = np.zeros((N, nbstepx_max))

# Computation of the stationnary states of the branch 1
stationary = stationary_state(0, np.array([Het0, Vet0]))
Hp[0, :nbstepx[0]], Vp[0, :nbstepx[0]] = stationary[:, 0], stationary[:, 1]

# Generation of the stationary states at the start of river branches 2, ..., N with Kirchoff's laws
for i in range(1, N-1):
    Hp[i, 0] = Hp[0, -1]
    Vp[i, 0] = weight[i-1]*Vp[0][-1]
Hp[N-1, 0] = Hp[0][-1]
Vp[N-1, 0] = Vp[0][-1]*(1 - np.sum(weight))

# Computation of the stationary states for the river branches 2,..., N :
for i in range(1, N):
    stationary = stationary_state(i, np.array([Hp[i, 0], Vp[i, 0]]))
    Hp[i, :nbstepx[i]], Vp[i, :nbstepx[i]] = stationary[:, 0], stationary[:, 1]

Lx = [np.linspace(0, L[i], nbstepx[i]) for i in range(N)]

print("Simulation of the stationary states : success")

# Initialization for disturbances
sol = np.zeros((N, nbstept, nbstepx_max, 2))
for l in range(N):
    addL1 = 0
    if(l >= 1):
        addL1 = Lx[0][-1] # length of the first branch

    sol[l, 0, :nbstepx[l], 0] = v0(Lx[l]+addL1) + h0(Lx[l]+addL1)*np.sqrt(g/np.array(Hp[l, :nbstepx[l]]))
    sol[l, 0, :nbstepx[l], 1] = v0(Lx[l]+addL1) - h0(Lx[l]+addL1)*np.sqrt(g/Hp[l, :nbstepx[l]])

print("Initialization of the disturbances : success")

# Numerical simulation
def scheme_step_upwind(nx, z, f, Vet, Het, bathy):
    updated_z2 = np.zeros(nx-1)
    for j in range(nx-1):
        updated_z2[j] = z[j][1] + dt * (lambda2(Vet[j], Het[j]) * ((z[j + 1][1] - z[j][1]) / dx) - gamma2(f, Vet[j], Het[j], bathy[j]) * z[j][0] - delta2(f, Vet[j], Het[j], bathy[j]) * z[j][1])
    return updated_z2

def scheme_step_downwind(nx, z, f, Vet, Het, bathy):
    updated_z1 = np.zeros(nx-1)
    for j in range(1, nx):
        updated_z1[j-1] = z[j][0] - dt * (lambda1(Vet[j], Het[j]) * ((z[j][0] - z[j-1][0]) / dx) + gamma1(f, Vet[j], Het[j], bathy[j]) * z[j][0] - delta1(f, Vet[j], Het[j], bathy[j]) * z[j][1])
    return updated_z1

for i in range(1, nbstept):

    # Controls & computation until the node
    sol[0][i][0][0] = Kc[0]*sol[0][i-1][0][1]
    sol[0, i, 1:nbstepx[0], 0] = scheme_step_downwind(nbstepx[0], sol[0][i-1], k[0], Vp[0], Hp[0], C[0])

    for l in range(1, N):
        sol[l][i][nbstepx[l]-1][1] = Kc[l]*sol[l][i-1][nbstepx[l]-1][0]
        sol[l, i, :(nbstepx[l]-1), 1] = scheme_step_upwind(nbstepx[l], sol[l][i-1], k[l], Vp[l], Hp[l], C[l])

    # Generation (Kirchoff...) & computation from the node
    sol[0][i][-1][1] = (1 - 2/N)*sol[0][i][-1][0] + (2/N)*np.sum(sol[1:, i, 0, 1])
    sol[0, i, :(nbstepx[0]-1), 1] = scheme_step_upwind(nbstepx[0], sol[0][i-1], k[0], Vp[0], Hp[0], C[0])

    for l in range(1, N):
        sol[l, i, 0, 0] = sol[l, i, 0, 1] + sol[0, i, -1, 0] - sol[0, i, -1, 1]
        sol[l, i, 1:nbstepx[l], 0] = scheme_step_downwind(nbstepx[l], sol[l][i-1], k[l], Vp[l], Hp[l], C[l])

print("Régime perturbé simulé avec succès")

# Change of variables (to get some physics)
h = np.zeros((N, nbstept, nbstepx_max))
v = np.zeros((N, nbstept, nbstepx_max))
H = np.zeros((N, nbstept, nbstepx_max))
V = np.zeros((N, nbstept, nbstepx_max))
Zf = np.zeros((N, nbstepx_max)) # height of the floor

# the floor is assumed linear for simplicity but it is easy to adapt for smooth enough floors
for l in range(N):
    Zf[l, :nbstepx[l]] = C[l]*Lx[l]

for i in range(nbstept):
    h[:, i] = (sol[:, i, :, 0] - sol[:, i, :, 1]) * np.sqrt(Hp / g) / 2
    v[:, i] = (sol[:, i, :, 0] + sol[:, i, :, 1]) / 2
    H[:, i] = Hp + Zf + h[:, i, :]
    V[:, i] = Vp + v[:, i, :]

print("COV : success")

### Display
# useful functions
def contours(i):
    xmin = Lx[i][0] - 5
    xmax = Lx[i][nbstepx[i] - 1] + 5
    hmin = np.min(H[i, :, :nbstepx[i]])
    hmax = np.max(H[i])
    espace = (hmax - hmin) / 20
    hmin -= espace
    hmax += espace
    return xmin, xmax, hmin, hmax

def pcontours(i, L):
    xmin = Lx[i][0] - 5
    xmax = Lx[i][nbstepx[i] - 1] + 5
    hmin = np.min(L[i, :nbstepx[i]])
    hmax = np.max(L[i])
    espace = (hmax - hmin) / 20
    hmin -= espace
    hmax += espace
    return xmin, xmax, hmin, hmax

def Display_GP(val_name, unit, Val):
    fig = plt.figure()  # initialize the figure
    axes = []
    axes.append(fig.add_subplot(N-1, 2, 1))
    xmin, xmax, hmin, hmax = pcontours(0, Val)
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(hmin, hmax)
    axes[0].plot(Lx[0], Val[0, :nbstepx[0]])
    axes[0].set_xlabel('x(m)')
    axes[0].set_ylabel(val_name + "1*(" + unit + ")")

    for i in range(1, N):
        axes.append(fig.add_subplot(N-1, 2, 2*i))
        xmin, xmax, hmin, hmax = pcontours(i, Val)
        axes[i].set_xlim(xmin, xmax)
        axes[i].set_ylim(hmin, hmax)
        axes[i].plot(Lx[i], Val[i, :nbstepx[i]])
        if(i == N - 1):
            axes[i].set_xlabel('x(m)')
        if(i >= 2):
            axes[i].set_ylabel(val_name + str(i+1) + "*(" + unit + ")")

    plt.show()

def Norm2(vals):
    return dx*(np.sum(np.sum(np.sum(vals**2))))

# stationary state
Display_GP('H', 'm', Hp) # Height
Display_GP('V', 'm/s', Vp) # Speed

# L2 error as a function of the time
L2 = [Norm2(sol[:, i]) for i in range(nbstept)]
Lt = [dt*i for i in range(nbstept)]
plt.title("Variation of the L2 norm of the perturbation")
plt.xlabel("time(s)")
plt.ylabel("L2 norm")
plt.plot(Lt, L2)
plt.show()

# Animation

fig = plt.figure()  # initialize the figure
axes = []
lines = []

axes.append(fig.add_subplot(N-1, 2, 1))
lines.append(axes[0].plot([], [], label='Flow 1')[0])
xmin, xmax, hmin, hmax = contours(0)
axes[0].set_xlim(xmin, xmax)
axes[0].set_ylim(hmin, hmax)
axes[0].set_xlabel('x(m)')
axes[0].set_ylabel('H1(m)')

for i in range(1, N):
    axes.append(fig.add_subplot(N-1, 2, 2*i))
    lines.append(axes[i].plot([], [], label='Flow')[0])
    xmin, xmax, hmin, hmax = contours(i)
    axes[i].set_xlim(xmin, xmax)
    axes[i].set_ylim(hmin, hmax)
    if(i == N - 1):
        axes[i].set_xlabel('x(m)')
    if(i >= 2):
        axes[i].set_ylabel("H" + str(i+1) + "(m)")

# function to define when blit=True
# create the background of the animation which will be on each image
def init():
    for i in range(N):
        lines[i].set_data(Lx[i], Zf[i])
    return lines


def animate(i):
    for j in range(N):
        lines[j].set_data(Lx[j], H[j, i, :nbstepx[j]])
    return lines


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nbstept, blit=True, interval=15, repeat=True)

print("Animation of the disturbances : success")

plt.show()

