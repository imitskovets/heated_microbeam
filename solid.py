import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def updatefig(i):
    im.set_array(Termo[i])
    return im,
alfa = 10
N_visual = 150                                                                          # kol shagov po dt Visual
P = 5                                                                                   # pokaz kartinki kazdie P kadrov
Y_full = 10 * alfa                                                                      # koll shag po dy
X_full = 20 * alfa                                                                      # koll shag po dx
rect_x_start = 3 * alfa
rect_x_end = 9 * alfa
rect_y_start = 2 * alfa
rect_y_end = 4 * alfa
T1 = 10
T0 = 2

interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000/interval, bitrate = 2000)
fig1 = plt.figure(1)

# begin conditions
VisualMap = np.ones((Y_full, X_full), dtype=float)
VisualMap = T0 * VisualMap
print(VisualMap)
for x in np.arange(X_full):
    for y in np.arange(Y_full):
        if (y <= rect_y_end) and (y >= rect_y_start) and (x >= rect_x_start) and (x <= rect_x_end):
            VisualMap[y, x] = T1
Termo = np.zeros(Y_full * X_full * N_visual).reshape(N_visual, Y_full, X_full)
Termo[0] = VisualMap
Termo[1] = VisualMap
#



#
setka = np.arange(Y_full * X_full).reshape(Y_full, X_full)
im = plt.imshow(np.sin(setka), cmap=plt.get_cmap('viridis'), animated=True)
plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(int(N_visual / P)), interval = interval)
#fig.show()
kino.save('VideoSolid.mp4', writer = writer)