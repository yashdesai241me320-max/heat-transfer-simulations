# heat-transfer-simulations

I'm a 3rd year Mechanical Engineering student at NIT Karnataka (Surathkal).

I built these simulations after coming across research on Inverse Heat Conduction
Problems -- specifically how you can estimate an unknown boundary heat flux just
from temperature sensor readings inside a material. Found this really interesting
because it connects numerical methods (which we did this semester) with real
experimental heat transfer problems.

These are my attempts at understanding the math from scratch before I could
confidently say I know what IHCP means.

---

## project 1 -- 1D transient heat conduction solver

Simulates temperature distribution in a steel rod over time using the
Finite Difference Method (explicit scheme).

**what it does:**
- solves the heat equation: dT/dt = alpha * d2T/dx2
- fixed temperature boundaries (Dirichlet BC)
- plots transient profiles at t = 0, 10, 30, 60, 120, 300 seconds
- validates against analytical steady-state solution
- computes heat flux using Fourier's law (q = -k dT/dx)

**what i learned building this:**
- had to keep the Fourier number r <= 0.5 for stability (blew up the first time)
- switched from forward difference to np.gradient for flux -- forward difference
  gave wrong values at the rod boundaries

![result](project1_heat_conduction.png)

---

## project 2 -- inverse heat conduction problem (IHCP)

Given temperature readings from 5 thermocouples along a rod, can we figure out
the heat flux that was applied at the boundary -- without measuring it directly?

**approach:**
- generate synthetic "experimental" data using the forward solver with known q
- add Gaussian noise (sigma = 0.5°C) to simulate real thermocouple error
- set up a least-squares optimisation problem using SciPy
- optimizer starts from a wrong guess (5000 W/m2) and converges to true value (~15000 W/m2)
- recovery error < 4% despite noisy sensor data

**this is directly related to IHCP research** -- the same principle is used in
real engineering problems where heat flux at a surface can't be measured directly
(like inside a combustion chamber or a nuclear reactor wall)

![result](project2_inverse_heat_transfer.png)

---

## how to run

```
pip install numpy matplotlib scipy
python heat_conduction_solver.py
python inverse_heat_transfer.py
```

tested on Python 3.10

---

## references
- Cengel & Ghajar, Heat and Mass Transfer (used for material properties and theory)
- our Numerical Methods course notes (FDM stability analysis)
