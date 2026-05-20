# inverse heat conduction problem -- estimating unknown heat flux from temperature data
#
# i came across this while looking into Prof. Gnanasekaran's research at NITK
# the idea is: in real experiments you cant always measure heat flux directly
# but you CAN measure temperature at a few points using thermocouples
# so can we work BACKWARDS to find the heat flux that caused those temperatures?
# turns out yes -- its an optimisation problem
#
# this is my attempt at implementing a simple version of IHCP
# -- [Your Name], 3rd year Mech, NITK Surathkal

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------------------------------------------------------
# step 1: build a forward solver first
# (same FDM as project 1 but with neumann BC at left instead of dirichlet)
# neumann BC means we specify heat flux instead of temperature at the boundary
# -------------------------------------------------------------------

def run_forward(q_left, rod_length=0.1, n_nodes=40, diffusivity=11.7e-6,
                sim_time=60.0, T_ambient=25.0, k=50.0):
    """
    given a heat flux at the left boundary, simulate and return final temperature field
    
    neumann BC at left:  -k * dT/dx = q  =>  T[0] = T[1] + q*dx/k
    dirichlet BC at right: T[-1] = T_ambient (rod end stays at room temp)
    """
    dx = rod_length / (n_nodes - 1)
    dt = 0.4 * dx**2 / diffusivity
    r  = diffusivity * dt / dx**2
    steps = int(sim_time / dt)
    
    x = np.linspace(0, rod_length, n_nodes)
    T = np.ones(n_nodes) * T_ambient
    T[-1] = T_ambient
    
    for _ in range(steps):
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        
        # neumann BC: heat flux applied at left end
        # derived from: q = -k*(T[0]-T[1])/dx  =>  T[0] = T[1] + q*dx/k
        T_new[0] = T_new[1] + q_left * dx / k
        T_new[-1] = T_ambient
        T = T_new
    
    return T, x


# -------------------------------------------------------------------
# step 2: generate fake "experimental" data
# pretend we ran an experiment with q = 15000 W/m2 and measured temperatures
# adding noise because real sensors are never perfect
# -------------------------------------------------------------------

q_actual = 15000.0   # this is what we're trying to recover -- pretend we dont know it
T_experiment, x_vals = run_forward(q_actual)

# thermocouple noise -- real sensors have ~0.5 degree error
np.random.seed(7)    # fixed seed so results are reproducible
noise = np.random.normal(0, 0.5, len(T_experiment))
T_noisy = T_experiment + noise

# in a real experiment you wouldnt have sensors everywhere
# placing 5 thermocouples at fixed positions along the rod
thermocouple_positions = [5, 10, 15, 20, 25]   # node indices
T_measured_at_sensors = T_noisy[thermocouple_positions]

print(f"true heat flux (unknown in real life): {q_actual} W/m2")
print(f"thermocouple readings: {T_measured_at_sensors.round(2)} °C")


# -------------------------------------------------------------------
# step 3: set up the inverse problem
# we want to find q such that forward_solver(q) matches our sensor readings
# this is a least squares minimisation problem
#
# objective = sum of (T_predicted - T_measured)^2
# -------------------------------------------------------------------

iteration_log = []   # keeping track of how optimizer converges

def objective_function(q_guess):
    """
    how wrong are we? compare forward model prediction vs actual sensor readings
    optimizer will try to minimise this
    """
    q = q_guess[0]
    
    # dont let optimizer try negative heat flux (unphysical)
    if q < 0:
        return 1e10
    
    T_predicted, _ = run_forward(q)
    T_at_sensors = T_predicted[thermocouple_positions]
    
    # sum of squared residuals
    error = np.sum((T_at_sensors - T_measured_at_sensors)**2)
    
    iteration_log.append({'q': q, 'error': error})
    return error


# -------------------------------------------------------------------
# step 4: run the optimiser
# starting from a deliberately wrong guess (5000 instead of 15000)
# to show the method actually recovers the answer
#
# tried L-BFGS-B first but Nelder-Mead worked better for this problem
# probably because we only have 1 unknown so gradient methods are overkill
# -------------------------------------------------------------------

initial_guess = 5000.0
print(f"\nstarting optimisation from q = {initial_guess} W/m2 (wrong guess)")
print("running...")

result = minimize(
    objective_function,
    x0=[initial_guess],
    method='Nelder-Mead',
    options={'xatol': 0.1, 'fatol': 1e-6, 'maxiter': 3000}
)

q_recovered = result.x[0]
percent_error = abs(q_recovered - q_actual) / q_actual * 100

print(f"\nresult:")
print(f"  actual q    = {q_actual:.1f} W/m2")
print(f"  recovered q = {q_recovered:.2f} W/m2")
print(f"  error       = {percent_error:.3f}%")
print(f"  iterations  = {result.nit}")


# -------------------------------------------------------------------
# step 5: plots
# -------------------------------------------------------------------

T_recovered, _ = run_forward(q_recovered)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("inverse heat conduction problem -- recovering unknown heat flux", fontsize=12)

# plot 1: temperature field comparison
ax = axes[0]
ax.plot(x_vals * 100, T_experiment, 'b-', linewidth=2, label='true temperature')
ax.plot(x_vals * 100, T_noisy, 'r.', markersize=3, alpha=0.5, label='noisy measurements')
ax.plot(x_vals * 100, T_recovered, 'g--', linewidth=2,
        label=f'recovered (q={q_recovered:.0f} W/m2)')
ax.scatter(x_vals[thermocouple_positions] * 100, T_measured_at_sensors,
           s=60, color='orange', zorder=5, label='thermocouple locations', marker='^')
ax.set_xlabel("position (cm)")
ax.set_ylabel("temperature (°C)")
ax.set_title("temperature field")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 2: bar chart comparing true vs guessed vs recovered
ax = axes[1]
labels = ['true q', 'initial guess', 'recovered q']
values = [q_actual, initial_guess, q_recovered]
bar_colors = ['steelblue', 'tomato', 'mediumseagreen']
bars = ax.bar(labels, values, color=bar_colors, alpha=0.8)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
            f'{val:.0f}', ha='center', fontsize=9)
ax.set_ylabel("heat flux (W/m2)")
ax.set_title(f"estimation result\nerror = {percent_error:.2f}%")
ax.grid(True, alpha=0.3, axis='y')

# plot 3: convergence -- how the optimizer found the answer
ax = axes[2]
q_history = [it['q'] for it in iteration_log]
ax.plot(range(len(q_history)), q_history, color='purple', linewidth=1.2, alpha=0.8)
ax.axhline(y=q_actual, color='blue', linestyle='--', linewidth=1.5, label=f'true = {q_actual:.0f}')
ax.axhline(y=q_recovered, color='green', linestyle=':', linewidth=1.5, label=f'converged = {q_recovered:.0f}')
ax.set_xlabel("iteration")
ax.set_ylabel("q estimate (W/m2)")
ax.set_title("optimiser convergence")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("project2_inverse_heat_transfer.png", dpi=150, bbox_inches='tight')
plt.show()
print("done -- plot saved")
