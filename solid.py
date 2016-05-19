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


# initiate constants
alfa = 1
N_visual = 150  # kol shagov po dt Visual
betta = 10
N_time = N_visual * betta
P = 5  # pokaz kartinki kazdie P kadrov
Y_full = 10 * alfa  # koll shag po dy
X_full = 20 * alfa  # koll shag po dx
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

# begin conditions
VisualMap = np.ones(((Y_full + 1), (X_full + 1)), dtype=float)
VisualMap = T0 * VisualMap
for x in np.arange(X_full):
    for y in np.arange(Y_full):
        if (y <= rect_y_end) and (y >= rect_y_start) and (x >= rect_x_start) and (x <= rect_x_end):
            VisualMap[y, x] = T1
Termo = np.zeros((Y_full + 1) * (X_full + 1) * N_visual).reshape(N_visual, (Y_full + 1), (X_full + 1))
F = np.zeros((Y_full + 1) * (X_full + 1) * N_time).reshape(N_time, (Y_full + 1), (X_full + 1))
F[0] = VisualMap
F[1] = VisualMap
print(F[1, Y_full - 1, X_full - 1])
# main calculus
for n in range(N_time - 2):
    # go through y
    n_i = n + 2
    for x in range(X_full - 1):
        for y in range(rect_y_start - 1):
            x_eff = x + 1
            y_eff = y + 1
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff + 1, x_eff],
                                            F[n_i - 1, y_eff - 1, x_eff])

    for x in range(rect_x_start - 1):
        for y in range(rect_y_end - rect_y_start - 1):
            x_eff = x + 1
            y_eff = y + rect_y_start + 1
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff + 1, x_eff],
                                            F[n_i - 1, y_eff - 1, x_eff])

    for x in range(X_full - rect_x_end - 1):
        for y in range(rect_y_end - rect_y_start - 1):
            x_eff = x + rect_x_end + 1
            y_eff = y + rect_y_start + 1
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff + 1, x_eff],
                                            F[n_i - 1, y_eff - 1, x_eff])

    for x in range(X_full - 1):
        for y in range(Y_full - rect_y_end - 1):
            x_eff = x + 1
            y_eff = y + rect_y_end + 1
            print(x_eff)
            print(y_eff)
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff + 1, x_eff],
                                            F[n_i - 1, y_eff - 1, x_eff])

    # go through x
    for y in range(Y_full - 1):
        for x in range(rect_x_start - 1):
            x_eff = x + 1
            y_eff = y + 1
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff, x_eff + 1],
                                            F[n_i - 1, y_eff, x_eff - 1])

    for y in range(rect_y_start - 1):
        for x in range(rect_x_end - rect_x_start - 1):
            x_eff = x + rect_x_start + 1
            y_eff = y + 1
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff, x_eff + 1],
                                            F[n_i - 1, y_eff, x_eff - 1])

    for y in range(Y_full - rect_y_end - 1):
        for x in range(rect_x_end - rect_x_start - 1):
            x_eff = x + rect_x_start + 1
            y_eff = y + rect_y_end + 1
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff, x_eff + 1],
                                            F[n_i - 1, y_eff, x_eff - 1])

    for y in range(Y_full - 1):
        for x in range(X_full - rect_x_end - 1):
            x_eff = x + rect_x_end + 1
            y_eff = y + 1
            F[n_i, y_eff, x_eff] = funcMain(F[n_i - 1, y_eff, x_eff], F[n_i - 1, y_eff, x_eff + 1],
                                            F[n_i - 1, y_eff, x_eff - 1])

# generate video array
for n in range(N_visual):
    Termo[n] = F[n * betta] / T1
# draw video
interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000 / interval, bitrate=2000)
fig1 = plt.figure(1)

setka = np.arange(Y_full * X_full).reshape(Y_full, X_full)
im = plt.imshow(np.sin(setka), cmap=plt.get_cmap('viridis'), animated=True)
plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(int(N_visual / P)), interval=interval)
# fig.show()
kino.save('VideoSolid.mp4', writer=writer)
