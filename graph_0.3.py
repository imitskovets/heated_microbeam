import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def updatefig(i):
    im.set_array(Termo[i])
    return im,
alfa = 10
N_visual = 150                                                                          # kol shagov po dt Visual
P = 5                                                                                   # pokaz kartinki kazdie P kadrov
Y_full = 10 * alfa                                                                             # koll shag po dy
X_full = 20 * alfa                                                                            # koll shag po dx
rect_x_start = 3 * alfa
rect_x_end = 7 * alfa
rect_y_start = 3 * alfa
rect_y_end = 5 * alfa

interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000/interval, bitrate = 2000)
fig1 = plt.figure(1)

#
VisualMap = np.zeros(Y_full * X_full).reshape(1, Y_full, X_full)
Termo = np.zeros(Y_full * X_full * N_visual).reshape(N_visual, Y_full, X_full)

#print(VisualMap)

pi = np.arccos(-1)
print(pi)
print(np.sin(2*3.14*3 / 5))

for n in np.arange(N_visual):
    for x in np.arange(X_full):
        for y in np.arange(Y_full):
            if x < rect_x_start:
                Termo[n, y, x] = np.sin(2 * pi * (x + n) / X_full) + np.cos(2 * pi *(y - n) / Y_full)
            if x > rect_x_end:
                Termo[n, y, x] = np.sin(2 * pi * (x + n) / X_full) + np.cos(2 * pi * (y - n) / Y_full)
            if y < rect_y_start:
                Termo[n, y, x] = np.sin(2 * pi * (x + n) / X_full) + np.cos(2 * pi * (y - n) / Y_full)
            if y > rect_y_end:
                Termo[n, y, x] = np.sin(2 * pi * (x + n) / X_full) + np.cos(2 * pi * (y - n) / Y_full)
            if (y <= rect_y_end) and (y >= rect_y_start) and (x >= rect_x_start) and (x <= rect_x_end):
                Termo[n, y, x] = 1
    Termo[n, 0, 0] = 1
print(Termo)
#

setka = np.arange(Y_full * X_full).reshape(Y_full, X_full)
#WTF????????????????
im = plt.imshow(np.sin(setka))

plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(int(N_visual / P)), interval = interval)
#fig.show()
kino.save('Video.mp4', writer = writer)
