import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def updatefig(i):
    im.set_array(Utime[i])
    return im,

def norm(u, nx, ny):
    sum = 0
    for i0 in range(nx - 1):
        for j0 in range(ny - 1):
            sum += u[i0][j0]**2
    res = sum**(1/2)
    return res

def diff_el(u1, u2, nx, ny):
    diff = np.zeros((nx, ny), dtype=float)
    for i0 in range(nx - 1):
        for j0 in range(ny):
            diff[i0][j0] = u1[i0][j0] - u2[i0][j0]
    return diff

def zeid(uip, uim, ujp, ujm, f):
    res = (uip + uim + ujp + ujm + h*h*f) / 4
    return res

def zeidI( uim, ujp, ujm, f):
    res = (ujp + uim + ujm + h*h*f) / 4
    return res
def zeidJ(uip, uim, ujm, f):
    res = (uip + uim + ujm + h*h*f) / 4
    return res
def one_iteration():
    for i0 in range(1, Nx - 1, 1):
        for j0 in range(1, Ny - 1, 1):
            U[i0][j0] = zeid(Utemp[i0 + 1][j0], U[i0 - 1][j0], Utemp[i0][j0 + 1], U[i0][j0 - 1], f[i0][j0])
    for i0 in range(1, Nx - 1 , 1):
        U[i0][Ny - 1] = zeidJ(Utemp[i0 + 1][Ny - 1], U[i0 - 1][Ny - 1], U[i0][Ny - 1 - 1], f[i0][Ny - 1])
    for j0 in range(1, Ny - 1, 1):
        U[Nx - 1][j0] = zeidI(U[Nx - 1 - 1][j0], Utemp[Nx - 1][j0 + 1], U[Nx - 1][j0 - 1], f[Nx - 1][j0])

Nx = 100
Ny = 100
hx = 0.1
hy = 0.1
h = hx
pi = np.arccos(-1)
accuracy = 0.01
iterations = 100
max = 0


U = np.zeros((Nx, Ny), dtype=float)
Utime = np.zeros(Ny * Nx * iterations).reshape(iterations, Nx, Ny)
f = np.zeros((Nx, Ny), dtype=float)
U0 = np.zeros((Nx, Ny), dtype=float)
Utemp = np.zeros((Nx, Ny), dtype=float)

for i in range(Nx):
    for j in range(Ny):
        f[i][j] = np.sin(pi*i*hx)*np.sin(pi*j*hy)
        U0[i][j] = np.sin(pi*i*hx)*np.sin(pi*j*hy) / (2 * pi**2)

for count in range(iterations - 1):
    one_iteration()
    accuracy = norm(diff_el(U, Utemp, Nx, Ny), Nx, Ny)
    Utime[count] = U - U0
    Utemp = U
    #print(accuracy)

#print(norm(diff_el(U, U0, Nx, Ny), Nx, Ny))
#print(Utemp)







# visual
interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000 / interval, bitrate=2000)
fig1 = plt.figure(1)
setka = np.arange(Nx * Ny).reshape(Nx , Ny)
im = plt.imshow(np.sin(setka)/ 25, cmap=plt.get_cmap('viridis'), animated=True)
plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(iterations), interval=interval)
#fig1.show()
kino.save('Ecliptic.mp4', writer=writer)