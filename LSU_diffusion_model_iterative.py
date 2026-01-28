import numpy as np
import matplotlib.pyplot as plt

D = 100
Lx = 300

dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)

C = np.zeros_like(x)
C_left = 500
C_right = 0
C[x <= Lx//2] = C_left
C[x > Lx//2] = C_right

plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial concentration profile")

nt = 5000
dt = 0.5 * dx ** 2 / D

dt

for t in range(0, nt):
	C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])

plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final concentration profile")

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

Lx = 300
D = 100
dx = 0.5
x = np.arange(0, Lx, dx)
nt = 5000

dt = 0.5 * dx**2 / D

display_interval = 5
n_steps_per_update = 100

C0 = np.zeros_like(x)
C_left, C_right = 500, 0
C0[x <= Lx/2] = C_left

C = C0.copy()
step = 0

step_button = widgets.Button(description="Step")
play = widgets.Play(interval=100, value=0, min=0, max=nt, step=1)
speed_slider = widgets.IntSlider(description="Speed (ms)", value=100, min=10, max=500, step=10)
last_step_button = widgets.Button(description="Jump to last step")
reset_button = widgets.Button(description="Reset")
out = widgets.Output()

def render():
    """Update the plot."""
    with out:
        out.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(x, C, lw=2)
        ax.set_ylim(0, 500)
        ax.set_xlabel("x")
        ax.set_ylabel("C")
        ax.set_title(f"Step {step}")
        plt.show()

def advance(n=1):
    """Advance the diffusion simulation by n steps."""
    global C, step
    for _ in range(n):
        if step >= nt:
            return
        C[1:-1] += D * dt / dx**2 * (C[:-2] - 2*C[1:-1] + C[2:])
        C[0], C[-1] = C_left, C_right
        step += 1

def on_step_clicked(b):
    advance(n_steps_per_update)
    if step % display_interval == 0 or step >= nt:
        render()

def on_play_change(change):
    advance(n_steps_per_update)
    if step % display_interval == 0 or step >= nt:
        render()

def on_speed_change(change):
    play.interval = change['new']

def on_last_step_clicked(b):
    advance(nt - step)
    render()

def on_reset_clicked(b):
    global C, step
    C[:] = C0
    step = 0
    render()

step_button.on_click(on_step_clicked)
play.observe(on_play_change, names='value')
speed_slider.observe(on_speed_change, names='value')
last_step_button.on_click(on_last_step_clicked)
reset_button.on_click(on_reset_clicked)

display(widgets.HBox([step_button, play, last_step_button, reset_button, speed_slider]), out)
render()
