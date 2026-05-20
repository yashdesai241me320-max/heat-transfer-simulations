# trying to simulate heat conduction in a metal rod
# started this after our numerical methods class when prof showed us FDM
# wanted to see if i could actually implement it myself for a steel rod
#
# ref: Cengel Heat Transfer textbook (chapter 5), and some notes from class
# -- [Your Name], 3rd year Mech, NITK Surathkal

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# rod properties -- using steel because thats what our HT textbook uses
# -------------------------------------------------------------------
L = 0.1          # rod length in meters (10 cm)
T_hot = 300.0    # left end kept at 300 degrees (like a furnace end)
T_cold = 25.0    # right end at room temp
T_start = 25.0   # whole rod starts at room temp
alpha = 11.7e-6  # thermal diffusivity of steel, got this from cengel appendix

# -------------------------------------------------------------------
# grid setup
# tried N=100 first but it was slow, N=50 works fine for now
# -------------------------------------------------------------------
N = 50
dx = L / (N - 1)

# stability condition for explicit FDM -- learned this in numerical methods
# r must be <= 0.5 otherwise solution blows up (found this out the hard way)
dt = 0.4 * dx**2 / alpha   # keeping r = 0.4 to stay safely under 0.5
r = alpha * dt / dx**2

# quick check
print(f"r (fourier number) = {r:.4f}  -- needs to be under 0.5")
print(f"dx = {dx*1000:.2f} mm,  dt = {dt:.4f} s")

total_time = 300.0  # simulate for 5 minutes
num_steps = int(total_time / dt)
print(f"total steps to simulate: {num_steps}")

# -------------------------------------------------------------------
# initialise temperature array
# -------------------------------------------------------------------
x = np.linspace(0, L, N)
T = np.ones(N) * T_start
T[0] = T_hot     # boundary condition: left end fixed at 300C
T[-1] = T_cold   # boundary condition: right end fixed at 25C

# -------------------------------------------------------------------
# time stepping -- explicit finite difference
# formula: T_new[i] = T[i] + r*(T[i+1] - 2*T[i] + T[i-1])
# this comes directly from discretising d2T/dx2
# -------------------------------------------------------------------

# saving snapshots at these times to plot later
save_at = [0, 10, 30, 60, 120, 300]
saved_profiles = {}

for step in range(num_steps):
    t_now = step * dt
    
    T_new = T.copy()
    T_new[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
    
    # enforce boundary conditions every step
    T_new[0] = T_hot
    T_new[-1] = T_cold
    
    T = T_new
    
    # save snapshot if we're close to one of our target times
    for t_snap in save_at:
        if abs(t_now - t_snap) < dt / 2:
            saved_profiles[t_snap] = T.copy()

# -------------------------------------------------------------------
# analytical steady state solution -- just a straight line from 300 to 25
# T(x) = T_hot + (T_cold - T_hot) * x/L
# using this to verify my numerical result makes sense
# -------------------------------------------------------------------
T_analytical = T_hot + (T_cold - T_hot) * x / L

# -------------------------------------------------------------------
# plotting
# NOTE: initially i only plotted temperature but added heat flux later
# because heat flux is what shows up in inverse problems
# heat flux q = -k * dT/dx   (fourier's law, k=50 for steel)
# -------------------------------------------------------------------
k_steel = 50.0  # W/mK

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("1D heat conduction in steel rod -- FDM simulation", fontsize=13)

# plot 1: transient profiles at different times
ax = axes[0]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(save_at)))
for i, t_s in enumerate(save_at):
    if t_s in saved_profiles:
        ax.plot(x * 100, saved_profiles[t_s], color=colors[i], linewidth=1.8,
                label=f't = {t_s}s')
ax.set_xlabel("position (cm)")
ax.set_ylabel("temperature (°C)")
ax.set_title("temperature profiles over time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 2: comparing final numerical result vs analytical steady state
# they should match -- if not something is wrong with my code
ax = axes[1]
ax.plot(x * 100, T, 'b-', linewidth=2, label='FDM at t=300s')
ax.plot(x * 100, T_analytical, 'r--', linewidth=1.5, label='analytical (steady state)')
ax.set_xlabel("position (cm)")
ax.set_ylabel("temperature (°C)")
ax.set_title("numerical vs analytical (steady state)")
ax.legend()
ax.grid(True, alpha=0.3)

# plot 3: heat flux using fourier's law
# NOTE: i used np.gradient here instead of manual forward difference
# forward difference was giving weird values at the boundaries
ax = axes[2]
dTdx = np.gradient(T, x)
q = -k_steel * dTdx
ax.plot(x * 100, q / 1000, 'g-', linewidth=2)
ax.fill_between(x * 100, q / 1000, alpha=0.15, color='green')
ax.set_xlabel("position (cm)")
ax.set_ylabel("heat flux (kW/m²)")
ax.set_title("heat flux distribution (q = -k dT/dx)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("project1_heat_conduction.png", dpi=150, bbox_inches='tight')
plt.show()
print("done -- plot saved")
