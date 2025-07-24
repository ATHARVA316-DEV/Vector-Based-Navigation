"""
central_complex_navigation_sim.py

Live simulation of the insect-like central-complex (CX) navigation model
described by Le Moël et al. (9).  Runs an agent that

1. explores to discover food sources,
2. stores vector memories when it finds food,
3. homes to the nest via path-integration,
4. returns to remembered food using stored vectors,
5. takes novel shortcuts between learned sites,
6. forages serially using nearest-food selection.

Visualization:
──────────────
Left panel   – 2-D arena with nest, food, trajectory, and heading arrows  
Right panels – three bar charts refreshed every frame showing  
               TB1 ring attractor (heading),  
               CPU4 path-integration (home vector),  
               CPU1 memory comparison signal.

Dependencies:
─────────────
Python ≥3.9, NumPy, Matplotlib.

Run:
────
$ pip install numpy matplotlib
$ python central_complex_navigation_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------  MODEL CONSTANTS  ---------------------------- #
N_COLS          = 16                   # columns in CX (TB1 & CPU4)
ARENA_SIZE      = 200                  # squared arena edge length
STEP_SIZE       = 1.0                  # distance moved each time step
MAX_STEPS       = 60_000               # simulation length
FOOD_LOCS       = np.array([[150, 50], # food A
                            [50, 150]])# food B
NEST            = np.array([100, 100]) # nest position
STORE_RANGE     = 4.0                  # radius for finding food / nest
PATH_DECAY      = 0.9998               # leakage for PI drift
RECALIB_RATE    = 0.001                # correction when arriving at nest
SEARCH_TURNS    = (-30, 30)            # deg, random turn range when exploring
MEM_REWARD      = 1.0                  # scaling stored CPU4 weights
SEED            = 2025                 # RNG seed for reproducibility
np.random.seed(SEED)

# ----------------------------  UTILITY FUNCS  ------------------------------ #
def wrap_angle(theta):
    """wrap radians to [-π, π)."""
    return (theta + np.pi) % (2*np.pi) - np.pi

def col_angles(n=N_COLS):
    """center angle of each CX column in radians (0 at +x)."""
    return np.linspace(-np.pi, np.pi, n, endpoint=False)

# ----------------------------  AGENT CLASS  -------------------------------- #
class CXAgent:
    def __init__(self):
        self.pos          = NEST.copy().astype(float)
        self.heading      = np.random.uniform(-np.pi, np.pi)    # rad
        self.pi_vector    = np.zeros(2)                         # accumulated Δx,Δy
        self.cpu4         = np.zeros(N_COLS)                    # PI encoding
        self.tb1          = np.zeros(N_COLS)                    # heading encoding
        self.mem_bank     = []                                  # list of CPU4 vecs
        self.target_id    = None                                # active memory
        self.cpu1         = np.zeros(N_COLS)                    # mem comparison
        self.traj         = [self.pos.copy()]                   # visited points

    # ---------- neural encodings ---------- #
    def update_tb1(self):
        angs = col_angles()
        self.tb1 = np.cos(angs - self.heading)                  # ring attractor

    def update_cpu4(self):
        angs = col_angles()
        dist = np.linalg.norm(self.pi_vector)
        phi  = np.arctan2(self.pi_vector[1], self.pi_vector[0])
        self.cpu4 = dist * np.cos(angs - phi)                   # amplitude=dist

    def recall_memory(self):
        if self.target_id is None:                             # nothing recalled
            self.cpu1[:] = 0.0
            return
        mem = self.mem_bank[self.target_id]
        self.cpu1 = mem - self.cpu4                            # comparison signal

    # ---------- behavioral controllers ---------- #
    def explore_turn(self):
        """random walk used until any memory exists."""
        turn_deg = np.random.uniform(*SEARCH_TURNS)
        self.heading = wrap_angle(self.heading + np.deg2rad(turn_deg))

    def steer_to_vector(self, vec):
        """turn toward supplied vector (home or memory)."""
        desired = np.arctan2(vec[1], vec[0])
        error   = wrap_angle(desired - self.heading)
        self.heading += 0.15 * error                           # proportional control

    # ---------- memory handling ---------- #
    def store_memory(self):
        self.mem_bank.append(self.cpu4.copy() * MEM_REWARD)
        if self.target_id is None:       # first memory becomes default target
            self.target_id = 0

    def select_nearest_food(self):
        """winner-take-all: choose memory whose vector magnitude is smallest."""
        if not self.mem_bank:
            return
        dists = [np.linalg.norm(mem) for mem in self.mem_bank]
        self.target_id = int(np.argmin(dists))

    # ---------- main step ---------- #
    def step(self):
        # choose behavior
        at_nest = np.linalg.norm(self.pos - NEST) < STORE_RANGE
        at_food = None
        for i, food in enumerate(FOOD_LOCS):
            if np.linalg.norm(self.pos - food) < STORE_RANGE:
                at_food = i
                break

        # reset at nest (recalibrate PI, choose next target)
        if at_nest:
            self.pi_vector[:] = 0
            self.update_cpu4()
            self.recall_memory()
            # gradient descent to zero comparison error
            for _ in range(10):
                self.pi_vector *= (1 - RECALIB_RATE)
                self.update_cpu4()
            self.select_nearest_food()

        # store new memory if food discovered first time
        if at_food is not None and at_food >= len(self.mem_bank):
            self.store_memory()

        # steering strategy
        if self.mem_bank:
            if at_food is not None:
                # after eating, head home
                self.target_id = None
            if self.target_id is None:
                # go home (use PI)
                vec = -self.pi_vector.copy()
                self.steer_to_vector(vec)
            else:
                # go to memory
                mem_vec = self.mem_bank[self.target_id].copy()
                # decode memory vector orientation + magnitude
                dist = np.linalg.norm(mem_vec)
                phi  = np.arctan2((mem_vec * -np.sin(col_angles())).sum(),
                                  (mem_vec *  np.cos(col_angles())).sum())
                mem_cart = dist * np.array([np.cos(phi), np.sin(phi)])
                vec = -(self.pi_vector - mem_cart)           # perceived displacement
                self.steer_to_vector(vec)
        else:
            self.explore_turn()

        # move forward
        delta = STEP_SIZE * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.pos += delta
        self.traj.append(self.pos.copy())

        # path integration update with decay
        self.pi_vector -= delta
        self.pi_vector *= PATH_DECAY

        # neural updates
        self.update_tb1()
        self.update_cpu4()
        self.recall_memory()

# ----------------------------  VISUALIZATION  ------------------------------ #
agent = CXAgent()
angs = col_angles()

fig = plt.figure(figsize=(10, 5))

# arena subplot
ax0 = fig.add_subplot(1, 2, 1)
ax0.set_xlim(0, ARENA_SIZE), ax0.set_ylim(0, ARENA_SIZE)
ax0.set_aspect('equal'), ax0.set_title('Spatial Navigation')
nest_scatter, = ax0.plot([], [], 'o', color='saddlebrown', markersize=10, label='Nest')
food_scatter  = ax0.scatter(FOOD_LOCS[:,0], FOOD_LOCS[:,1],
                            color='green', s=80, label='Food')
trail_line,   = ax0.plot([], [], color='steelblue', lw=1)
agent_dot,    = ax0.plot([], [], 'o', color='red')
heading_arrow = ax0.arrow(0, 0, 0, 0, head_width=3, fc='red', ec='red')

# neural subplots
ax1 = fig.add_subplot(3, 2, 2)
ax2 = fig.add_subplot(3, 2, 4)
ax3 = fig.add_subplot(3, 2, 6)
for a, t in zip([ax1, ax2, ax3],
                ['TB1 Compass', 'CPU4 Path-Integration', 'CPU1 Memory Δ']):
    a.set_ylim(-N_COLS, N_COLS), a.set_xticks([]), a.set_title(t)
tb1_bar = ax1.bar(range(N_COLS), agent.tb1, color='orchid')
cpu4_bar = ax2.bar(range(N_COLS), agent.cpu4, color='gold')
cpu1_bar = ax3.bar(range(N_COLS), agent.cpu1, color='skyblue')

def update(frame):
    agent.step()

    # update arena
    xdata, ydata = np.array(agent.traj).T
    trail_line.set_data(xdata, ydata)
    agent_dot.set_data(agent.pos[0], agent.pos[1])

    # redrawing heading & PI vector arrows
    for art in ax0.patches: art.remove()
    heading = STEP_SIZE*7*np.array([np.cos(agent.heading), np.sin(agent.heading)])
    pi_cart = -agent.pi_vector
    arr1 = ax0.arrow(agent.pos[0], agent.pos[1], heading[0], heading[1],
                     head_width=2, fc='red', ec='red')
    arr2 = ax0.arrow(agent.pos[0], agent.pos[1], pi_cart[0], pi_cart[1],
                     head_width=2, fc='orange', ec='orange')
    ax0.patches.extend([arr1, arr2])

    # neural plots
    for rect, h in zip(tb1_bar, agent.tb1):  rect.set_height(h)
    for rect, h in zip(cpu4_bar, agent.cpu4): rect.set_height(h)
    for rect, h in zip(cpu1_bar, agent.cpu1): rect.set_height(h)

    return (trail_line, agent_dot, *tb1_bar, *cpu4_bar, *cpu1_bar)

ani = FuncAnimation(fig, update, frames=MAX_STEPS, interval=15, blit=True)
plt.tight_layout()
plt.show()
