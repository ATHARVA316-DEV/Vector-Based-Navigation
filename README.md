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
8. [Contributing](#contributing)
9. [License](#license)
10. [Citation](#citation)

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

> **Tip 💡** To share the demo online, host the repository with GitHub Pages or any static server.

---

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

Add any screenshots, GIF recordings or architecture diagrams to **`docs/media/`** and reference them with a relative path, e.g.
```md
![Trajectory](docs/media/trajectory.gif)
```
GitHub automatically shows the image in the README.

---

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

> Replace the placeholders below with your own captures.

| Trajectory & Neural Activity | Web GUI |
|------------------------------|---------|
| ![Demo](docs/media/demo_traj.gif) | ![GUI](docs/media/gui.png) |

---

## Research Background
* Model architecture follows the CX vector model by Le Moël *et al.* 2019 [44].
* Ring-attractor compass derives from CX head-direction circuits characterised by Heinze 2014 [27].
* CPU4 path-integrator design aligns with principles of insect PI summarised in Heinze *et al.* 2018 [34].

See `central-complex-algorithm.md` for a full literature review.

---

## Contributing
Pull requests are welcome!  Feel free to open issues for feature requests, bug reports or research discussions.  Please format code with **black** and include docstrings.

---

## License

[MIT](LICENSE) © 2025 Atharva Kulkarni & Contributors

---

## Citation
If you use this codebase in academic work, please cite:
```text
Kulkarni A. 2025. Vector-Based Navigation: Central Complex as a Potential Substrate for Vector Navigation. GitHub repository. https://github.com/ATHARVA316-DEV/Vector-Based-Navigation
```
