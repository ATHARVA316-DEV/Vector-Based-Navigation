# Vector-Based Navigation
The Central Complex as a Potential Substrate for Vector Based Navigation: A Comprehensive Explanation
*A biologically inspired toolkit and interactive demo of insect-style vector navigation built on Central Complex neural circuitry.*

<p align="center">
  <img src="docs/media/heading_ring.gif" alt="Ring-attractor compass" width="550"/>
</p>

---

## Table&nbsp;of&nbsp;Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Repository Layout](#repository-layout)
5. [Usage Examples](#usage-examples)
6. [Screenshots &amp; Media](#screenshots--media)
7. [Research Background](#research-background)

---

## Overview
Vector-Based Navigation is a research-grade simulation and visualisation suite that reproduces the **Central Complex (CX)** navigation circuits found in ants, bees and flies.  By combining a ring-attractor compass (TB1), path-integration neurons (CPU4) and vector-memory neurons (CPU1) the model can:

* integrate outbound paths and home directly to the nest,
* store food‐ward vectors, recall them on demand, and
* take novel shortcuts between remembered sites [44][29].

The project ships with both:

* **Python notebooks / scripts** for high-fidelity experiments, and
* an **HTML + JS web demo** for instant, interactive exploration in the browser.

---

## Key Features

| Module | Description |
|--------|-------------|
| **Ring-Attractor Compass (TB1)** | 8-cell sinusoidal head-direction network with noise injection |
| **Path-Integrator (CPU4)** | 16-cell distributed integrator encoding distance & direction |
| **Vector-Memory (CPU1)** | Reward-gated storage & recall of multiple goal vectors |
| **Behavioural Finite-State-Machine** | Exploration → Homing → Memory Return → Shortcutting → Route optimisation |
| **Live Visualisation** | Real-time plots of spatial trajectory & neural activity (Matplotlib / D3) |
| **Parameter Hooks** | Change gain, decay, neuron count, noise σ, speed etc. at runtime |
| **Web GUI** | Sliders for arena size, memories, mode switching & preset scenarios |

---

## Quick Start

### 1 – Clone & Install
```bash
# clone
$ git clone https://github.com/ATHARVA316-DEV/Vector-Based-Navigation.git
$ cd Vector-Based-Navigation

# create env (optional)
$ python -m venv .venv && source .venv/bin/activate

# install python deps
$ pip install -r requirements.txt  # numpy matplotlib
```

### 2 – Run the Python Simulation
```bash
$ python central_complex_navigation.py            # basic version
$ python central_complex_navigation_improved.py   # modular, extended
```
A matplotlib window will open showing the arena, neuron bars and state machine.

### 3 – Launch the Browser Demo
Simply open `index.html` in your favourite browser.
```bash
$ xdg-open index.html   # Linux
$ open index.html        # macOS
```
Use the side panel to adjust speed, number of food sources, PI decay etc.

## Repository Layout

```
Vector-Based-Navigation/
├── central_complex_navigation.py            # core simulation
├── central_complex_navigation_improved.py   # modular, extensible version
├── central_complex_navigation_sim.py        # CLI batch simulator
├── app.js  | index.html | style.css         # web interface
├── central-complex-algorithm.md             # algorithmic deep-dive
├── docs/
│   └── media/                               # ← put images & GIFs here
└── README.md
```

## Usage Examples

```python
from central_complex_navigation_improved import CentralComplexNavigator

# high-speed ant simulator
nav = CentralComplexNavigator(env_size=250, dt=0.1,
                              cpu4_gain=1.2, cpu4_decay=0.01,
                              cpu1_interference=0.02)
nav.speed = 2.0      # units / s
nav.run_simulation(steps=1200)
```

For batch experiments (e.g. parameter sweeps) use `central_complex_navigation_sim.py` which saves CSV logs of performance metrics.

---

## Screenshots & Media
index.html,style.css,app.js(web)
<img width="1915" height="972" alt="image" src="https://github.com/user-attachments/assets/4b980c9d-8e44-4136-b0ea-a59f8e818e18" />
Central Complex Navigation System (Exploration Enhanced)
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/14bd46c1-a66b-40d2-bfdd-31372f21ac40" />
Central Complex Navigation System (Improved)
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/6e62e7dd-838e-49bb-b630-b2055487b0d8" />
Central Complex Navigation System
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/45c0e066-d8ed-4ef8-88a9-df57dbddfc1d" />


## Research Background
* Model architecture follows the CX vector model by Le Moël *et al.* 2019 [44].
* Ring-attractor compass derives from CX head-direction circuits characterised by Heinze 2014 [27].
* CPU4 path-integrator design aligns with principles of insect PI summarised in Heinze *et al.* 2018 [34].

See `central-complex-algorithm.md` for a full literature review.

---

