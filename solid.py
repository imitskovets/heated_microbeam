import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def updatefig(i):
    im.set_array(Termo[i])
    return im,

def alfa_x_go():
    for i in range(X_full - 2):
        alfa_x[i + 1] = Ax / (Bx - Cx * alfa_x[i])

def alfa_y_go():
    for i in range(Y_full - 2):
        alfa_y[i + 1] = Ay / (By - Cy * alfa_y[i])

def betta_x_go(F, k):
    for i in range(X_full - 2):
        betta_x[i + 1] = (Cx * betta_x[i] - F[k][i]) / (Bx - Cx * alfa_x[i])

def betta_y_go(F, k):
    for i in range(Y_full - 2):
        betta_y[i + 1] = (Cy * betta_y[i] - F[i][k]) / (By - Cy * alfa_y[i])

def alfa_beam_x_1_go():
    for i in range(rect_x_start - 1):
        alfa_beam_x_1[i + 1] = Ax / (Bx - Cx * alfa_beam_x_1[i])

def alfa_beam_y_1_go():
    for i in range(rect_y_start - 1):
        alfa_beam_y_1[i + 1] = Ay / (By - Cy * alfa_beam_y_1[i])

def betta_beam_x_1_go(F, k):
    for i in range(rect_x_start - 1):
        betta_beam_x_1[i + 1] = (Cx * betta_beam_x_1[i] - F[k][i]) / (Bx - Cx * alfa_beam_x_1[i])

def betta_beam_y_1_go(F, k):
    for i in range(rect_y_start - 1):
        betta_beam_y_1[i + 1] = (Cy * betta_beam_y_1[i] - F[i][k]) / (By - Cy * alfa_beam_y_1[i])

def alfa_beam_x_2_go():
    for i in range(rect_x_end, X_full - 2, 1):
        alfa_beam_x_2[i + 1] = Ax / (Bx - Cx * alfa_beam_x_2[i])

def alfa_beam_y_2_go():
    for i in range(rect_y_end, Y_full - 2, 1):
        alfa_beam_y_2[i + 1] = Ay / (By - Cy * alfa_beam_y_2[i])

def betta_beam_x_2_go(F, k):
    for i in range(rect_x_end, X_full - 2, 1):
        betta_beam_x_2[i + 1] = (Cx * betta_beam_x_2[i] - F[k][i]) / (Bx - Cx * alfa_beam_x_2[i])

def betta_beam_y_2_go(F, k):
    for i in range(rect_y_end, Y_full - 2, 1):
        betta_beam_y_2[i + 1] = (Cy * betta_beam_y_2[i] - F[i][k]) / (By - Cy * alfa_beam_y_2[i])

# initiate constants
scale = 10
N_visual = 100  # kol shagov po dt Visual
jump = 1
N_time = N_visual * jump
P = 5  # pokaz kartinki kazdie P kadrov
Y_full = 15 * scale  # koll shag po dy
X_full = 11 * scale  # koll shag po dx
rect_x_start = 3 * scale
rect_x_end = 6 * scale
rect_y_start = 4 * scale
rect_y_end = 8 * scale
T1 = 10
T0 = 2
tau = 0.1
hx = 0.1
hy = 0.1
gamma = 0.1
kappa_x = tau * gamma / hx**2
kappa_y = tau * gamma / hy**2
Ax = kappa_x
Bx = 2 * kappa_x + 1
Cx = kappa_x
Ay = kappa_y
By = 2 * kappa_y + 1
Cy = kappa_y


# begin conditions
# check sizes
Termo = np.ones(Y_full * X_full * N_visual).reshape(N_visual, Y_full, X_full)
T = T0 * np.ones(Y_full * X_full * N_time).reshape(N_time, Y_full, X_full )
Ttemporal = np.ones(Y_full * X_full).reshape(Y_full, X_full)
Tmap = np.ones(Y_full * X_full).reshape(Y_full, X_full)

alfa_x = np.zeros((X_full), dtype=float)
betta_x = T0 * np.ones((X_full), dtype=float)
alfa_y = np.zeros((Y_full), dtype=float)
betta_y = T0 * np.ones((Y_full), dtype=float)

alfa_beam_x_1 = np.zeros((X_full), dtype=float)
betta_beam_x_1 = T0 * np.ones((X_full), dtype=float)
alfa_beam_y_1 = np.zeros((Y_full), dtype=float)
betta_beam_y_1 = T0 * np.ones((Y_full), dtype=float)

alfa_beam_x_2 = np.zeros((X_full), dtype=float)
betta_beam_x_2 = T0 * np.ones((X_full), dtype=float)
alfa_beam_y_2 = np.zeros((Y_full), dtype=float)
betta_beam_y_2 = T0 * np.ones((Y_full), dtype=float)


for x in np.arange(rect_x_end - rect_x_start + 1):
    for y in np.arange(rect_y_end - rect_y_start + 1):
        x_eff = x + rect_x_start
        y_eff = y + rect_y_start
        T[0, y_eff, x_eff] = T1
Ttemporal = T[0]
Tmap = T[0]

#print(T[0])

# main calculus
alfa_x_go()
alfa_y_go()
alfa_beam_x_1_go()
alfa_beam_x_2_go()
alfa_beam_y_1_go()
alfa_beam_y_2_go()
for n in range(N_time - 2):
    # go through x
    # check T and Ttemporal
    Ttemporal = Tmap
    for j in range(1, rect_y_start, 1):
        for i in range(X_full - 2, 0, -1):
            betta_x_go(-T[n], j)
            Ttemporal[j][i] = alfa_x[i] * Ttemporal[j][i + 1] + betta_x[i]
    for j in range(rect_y_start, rect_y_end + 1, 1):
        for i in range(rect_x_start - 1, 0, -1):
            betta_beam_x_1_go(-T[n], j)
            Ttemporal[j][i] = alfa_beam_x_1[i] * Ttemporal[j][i + 1] + betta_beam_x_1[i]
    for j in range(rect_y_start, rect_y_end + 1, 1):
        for i in range(X_full - 2, rect_x_end, -1):
            betta_beam_x_2_go(-T[n], j)
            Ttemporal[j][i] = alfa_beam_x_2[i] * Ttemporal[j][i + 1] + betta_beam_x_2[i]
    for j in range(rect_y_end + 1, Y_full - 2, 1):
        for i in range(X_full - 2, 0, -1):
            betta_x_go(-T[n], j)
            Ttemporal[j][i] = alfa_x[i] * Ttemporal[j][i + 1] + betta_x[i]
    # go through y
    for i in range(1, rect_x_start, 1):
        for j in range(X_full - 2, 0, -1):
            betta_y_go(-Ttemporal, j)
            T[n + 1][i][j] = alfa_y[i] * T[n + 1][i + 1][j] + betta_y[i]
    for i in range(rect_x_start, rect_x_end + 1, 1):
        for j in range(rect_x_start - 1, 0, -1):
            betta_beam_y_1_go(-Ttemporal, j)
            T[n + 1][i][j] = alfa_beam_y_1[i] * T[n + 1][i + 1][j] + betta_beam_y_1[i]
    for i in range(rect_x_start, rect_x_end + 1, 1):
        for j in range(X_full - 2, rect_x_end, -1):
            betta_beam_y_2_go(-Ttemporal, j)
            T[n + 1][i][j] = alfa_beam_y_2[i] * T[n + 1][i + 1][j] + betta_beam_y_2[i]
    for i in range(rect_x_end + 1, X_full - 2, 1):
        for j in range(X_full - 2, 0, -1):
            betta_y_go(-Ttemporal, j)
            T[n + 1][i][j] = alfa_y[i] * T[n + 1][i + 1][j] + betta_y[i]

#print(T[900])
# generate video array
for n in range(N_visual):
    Termo[n] = T[n * jump] / T1
# draw video
interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000 / interval, bitrate=2000)
fig1 = plt.figure(1)

setka = np.arange(Y_full * X_full).reshape(Y_full, X_full)
im = plt.imshow(np.sin(setka), cmap=plt.get_cmap('viridis'), animated=True)
plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(int(N_visual / P)), interval=interval)
#fig1.show()
kino.save('VideoSolid.mp4', writer=writer)
