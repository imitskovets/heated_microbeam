import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def updatefig(i):
    im.set_array(VisualMap[i])
    return im,

N = 1500                                                                                # kol shagov po dt
P = 5                                                                                   # pokaz kartinki kazdie P kadrov
Y_full = 10                                                                             # koll shag po dy
X_full = 15                                                                             # koll shag po dx

interval = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1000/interval, bitrate = 2000)
fig1 = plt.figure(1)

#
VisualMap = np.zeros(Y_full * X_full).reshape(1, Y_full, X_full)
Termo = np.arange(3 * 2 * Y_full * X_full).reshape(3, 2 * Y_full, X_full)

for n in np.arange(N):
    now = n%3
    if n in np.arange(0, N, P):
        VisualMap = np.vstack((VisualMap, Termo[now, ::2, :].reshape(1, Y_full, X_full)))
print(Termo[1])
#

setka = np.arange(Y_full * X_full).reshape(Y_full, X_full)
im = plt.imshow(np.sin(setka))

plt.figure(1)
kino = animation.FuncAnimation(fig1, updatefig, np.arange(int(N/P)), interval = interval)
#fig.show()
kino.save('Video.mp4', writer = writer)