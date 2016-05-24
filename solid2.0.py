import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# i - y | j - x

def updatefig(i):
    im.set_array(Termo[i])
    return im,


def progon(U, N, bound_x_1, bound_x_2, bound_y_1, bound_y_2):
    if bound_x_1 <= bound_x_2:
        x_step = 1
    else:
        x_step = - 1
    if bound_y_1 <= bound_y_2:
        y_step = 1
    else:
        y_step = - 1
    alpha_x = np.zeros((X_full), dtype=float)
    #beta_x = T0 * np.ones((X_full, abs(bound_y_2 - bound_y_1) - 1), dtype=float)
    beta_x = T0 * np.ones((X_full, Y_full), dtype=float)
    alpha_y = np.zeros((Y_full), dtype=float)
    #beta_y = T0 * np.ones((Y_full, abs(bound_x_2 - bound_x_1) - 1), dtype=float)
    beta_y = T0 * np.ones((Y_full, X_full), dtype=float)

    for i in range(bound_x_1, bound_x_2, x_step):
        alpha_x[i + 1] = A / (B - C * alpha_x[i])

    for i in range(bound_y_1, bound_y_2, y_step):
        alpha_y[i + 1] = A / (B - C * alpha_y[i])

    for i in range(bound_y_1 + 1, bound_y_2, y_step):
        for j in range(bound_x_1, bound_x_2 - 1, x_step):
            beta_x[j + 1][i] = (C * beta_x[j][i] - U[N][i][j]) / (B - C * alpha_x[j])
    if N == 0:
        print(beta_x)

    Ttemporal = U[N]
    # go through x
    for i in range(bound_y_1 + 1, bound_y_2, y_step):
        for j in range(bound_x_2 - 1, bound_x_1, - x_step):
            Ttemporal[i][j] = alpha_x[j] * Ttemporal[i][j + 1] + beta_x[j][i]

    for j in range(bound_x_1 + 1, bound_x_2, x_step):
        for i in range(bound_y_1, bound_y_2, y_step):
            beta_y[i + 1][j] = (C * beta_y[i][j] - Ttemporal[i][j]) / (B - C * alpha_y[i])

    # go through y
    for i in range(bound_x_1 + 1, bound_x_2, x_step):
        for j in range(bound_y_2 - 1, bound_y_1, - y_step):
            U[N + 1][j][i] = alpha_y[j] * U[N + 1][j + 1][i] + beta_y[j][i]

scale = 10
N_visual = 100  # kol shagov po dt Visual
jump = 1
N_time = N_visual * jump
P = 5  # pokaz kartinki kazdie P kadrov
Y_full = 13 * scale  # koll shag po dy
X_full = 10 * scale  # koll shag po dx
rect_x_start = 3 * scale
rect_x_end = 6 * scale
rect_y_start = 4 * scale
rect_y_end = 8 * scale
T1 = 10
T0 = 1
tau = 0.1
h = 0.1
gamma = 0.000001
kappa = tau * gamma / h**2
A = kappa
B = 2 * kappa + 1
C = kappa

# begin conditions
Termo = np.ones(Y_full * X_full * N_visual).reshape(N_visual, Y_full, X_full)
T = T0 * np.ones(Y_full * X_full * N_time).reshape(N_time, Y_full, X_full )
Ttemporal = np.ones(Y_full * X_full).reshape(Y_full, X_full)
Tmap = np.ones(Y_full * X_full).reshape(Y_full, X_full)

for n in range(N_time - 2):
    for x in np.arange(rect_x_start, rect_x_end + 1 , 1):
        for y in np.arange( rect_y_start, rect_y_end + 1, 1):
            T[n][y][x] = T1
Tmap = T[0]
print(T[0])

for n in range(N_time - 2):
    progon(T, n, rect_x_end, X_full - 1, rect_y_start, rect_y_end)
    progon(T, n, rect_x_start, 0, rect_y_start, rect_y_end)
    progon(T, n, rect_x_start, rect_x_end, rect_y_start, 0)
    progon(T, n, rect_x_start, rect_x_end, rect_y_end, Y_full - 1)
    progon(T, n, rect_x_start, 0, rect_y_start, 0)
    progon(T, n, rect_x_end, X_full - 1, rect_y_start, 0)
    progon(T, n, rect_x_start, 0, rect_y_end, Y_full - 1)
    progon(T, n, rect_x_end, X_full - 1, rect_y_end, Y_full - 1)
# generate video array
for n in range(N_visual-2):
    Termo[n] = (T[n * jump] - 0*Tmap) / T1
# draw video
interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000 / interval, bitrate=2000)
fig1 = plt.figure(1)

setka = np.arange(Y_full * X_full).reshape(Y_full, X_full)
im = plt.imshow(np.sin(setka)**2, cmap=plt.get_cmap('viridis'), animated=True)
#afmhot gist_heat viridis hot plasma

plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(int(N_visual)), interval=interval)
#fig1.show()
kino.save('VideoSolid.mp4', writer=writer)