import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def updatefig(i):
    im.set_array(Termo[i])
    return im,

def funcTildaPlus(f, f_plus, f_minus):
    res = 0
    if ksi > 0:
        res = f + (1 - gamma) * limiter(f, f_plus, f_minus) / 2
    if ksi < 0:
        res = f_plus - (1 - gamma) * limiter(f, f_plus, f_minus) / 2
    return res

def funcTildaMinus(f, f_plus, f_minus):
    res = 0
    if ksi > 0:
        res = f - (1 - gamma) * limiter(f, f_plus, f_minus) / 2
    if ksi < 0:
        res = f_plus + (1 - gamma) * limiter(f, f_plus, f_minus) / 2
    return res

def limiter(f, f_plus, f_minus):
    chis = (f_plus - f) * (f - f_minus)
    res = 0
    if chis > 0:
        res = 2 * chis / (f_plus - f_minus)
    return res

def funcMain(f, f_plus, f_minus):
    next_f = f - kappa * (funcTildaPlus(f, f_plus, f_minus) + funcTildaMinus(f, f_plus, f_minus))
    return next_f
alfa = 10
N_visual = 150                                                                          # kol shagov po dt Visual
N_time = N_visual * 10
P = 5                                                                                   # pokaz kartinki kazdie P kadrov
Y_full = 10 * alfa                                                                      # koll shag po dy
X_full = 20 * alfa                                                                      # koll shag po dx
rect_x_start = 3 * alfa
rect_x_end = 9 * alfa
rect_y_start = 2 * alfa
rect_y_end = 4 * alfa
T1 = 10
T0 = 2
ksi = 1
tau = 0.01
h = 0.1
kappa = ksi * tau / h
gamma = abs(kappa)

interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000/interval, bitrate = 2000)
fig1 = plt.figure(1)

# begin conditions
VisualMap = np.ones((Y_full, X_full), dtype=float)
VisualMap = T0 * VisualMap
for x in np.arange(X_full):
    for y in np.arange(Y_full):
        if (y <= rect_y_end) and (y >= rect_y_start) and (x >= rect_x_start) and (x <= rect_x_end):
            VisualMap[y, x] = T1
Termo = np.zeros(Y_full * X_full * N_visual).reshape(N_visual, Y_full, X_full)
F = np.zeros(Y_full * X_full * N_visual).reshape(N_visual, Y_full, X_full)
F[0] = VisualMap
F[1] = VisualMap
#



#
setka = np.arange(Y_full * X_full).reshape(Y_full, X_full)
im = plt.imshow(np.sin(setka), cmap=plt.get_cmap('viridis'), animated=True)
plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(int(N_visual / P)), interval = interval)
#fig.show()
kino.save('VideoSolid.mp4', writer = writer)