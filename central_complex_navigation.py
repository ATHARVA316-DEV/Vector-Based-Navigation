"""
Central Complex Navigation System
================================

A biologically inspired simulation of insect vector-based navigation, based on Le Moël et al. (2019).
Implements neural circuits (TB1, CPU4, CPU1), path integration, vector memory, and real-time visualization.

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# -----------------------------
# Neural Circuit Classes
# -----------------------------

class TB1Compass:
    """
    TB1 Compass Neurons (Ring Attractor)
    8 neurons encoding heading direction using sinusoidal activity.
    """
    def __init__(self, noise_std=0.05):
        self.n = 8
        self.angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.activity = np.zeros(self.n)
        self.noise_std = noise_std

    def update(self, heading_rad):
        # Sinusoidal encoding of heading
        self.activity = np.cos(self.angles - heading_rad)
        self.activity += np.random.normal(0, self.noise_std, self.n)
        self.activity = np.clip(self.activity, -1, 1)
        return self.activity

class CPU4Integrator:
    """
    CPU4 Path Integrator Neurons
    16 neurons encoding the home vector using distributed activity.
    """
    def __init__(self, noise_std=0.05):
        self.n = 16
        self.angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.activity = np.zeros(self.n)
        self.noise_std = noise_std
        self.gain = 1.0

    def integrate(self, delta_x, delta_y):
        # Project movement onto each neuron's preferred direction
        movement_angle = np.arctan2(delta_y, delta_x)
        movement_mag = np.hypot(delta_x, delta_y)
        self.activity += self.gain * movement_mag * np.cos(self.angles - movement_angle)
        self.activity += np.random.normal(0, self.noise_std, self.n)
        # Optional: decay or normalization
        return self.activity

    def get_vector(self):
        # Decode vector from distributed activity
        x = np.sum(self.activity * np.cos(self.angles))
        y = np.sum(self.activity * np.sin(self.angles))
        return x, y

    def reset(self):
        self.activity = np.zeros(self.n)

class CPU1Memory:
    """
    CPU1 Memory Neurons
    16 neurons for storing and recalling vector memories (e.g., food locations).
    Implements simple synaptic plasticity for memory storage.
    """
    def __init__(self, n_memories=4):
        self.n = 16
        self.n_memories = n_memories
        self.memories = []  # List of (activity, strength)

    def store(self, cpu4_activity, reward_strength=1.0):
        # Store a copy of the current CPU4 activity as a memory
        if len(self.memories) >= self.n_memories:
            self.memories.pop(0)  # Remove oldest if at capacity
        self.memories.append((cpu4_activity.copy(), reward_strength))

    def recall(self, memory_id=0):
        # Return the stored activity pattern for the given memory
        if 0 <= memory_id < len(self.memories):
            return self.memories[memory_id][0]
        else:
            return np.zeros(self.n)

    def get_memory_strengths(self):
        return [s for _, s in self.memories]

# -----------------------------
# Central Complex Navigator
# -----------------------------

class CentralComplexNavigator:
    """
    Main class for the Central Complex Navigation System.
    Handles neural updates, movement, memory, and visualization.
    """
    def __init__(self, env_size=100, dt=0.2):
        # Simulation parameters
        self.env_size = env_size
        self.dt = dt
        self.position = np.array([env_size/2, env_size/2], dtype=float)
        self.heading = random.uniform(0, 2*np.pi)
        self.speed = 1.0
        self.state = 'exploration'  # Behavioral state
        self.trajectory = [self.position.copy()]
        self.food_locations = []
        self.nest_location = np.array([env_size/2, env_size/2])
        self.current_target = None
        self.time = 0.0
        self.phase = 1
        # Neural circuits
        self.tb1 = TB1Compass()
        self.cpu4 = CPU4Integrator()
        self.cpu1 = CPU1Memory()
        # Visualization
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        plt.tight_layout()
        self.anim = None

    def update_neural_activity(self, movement, sensory_input=None):
        # Update TB1 compass
        tb1_activity = self.tb1.update(self.heading)
        # Update CPU4 path integrator
        self.cpu4.integrate(movement[0], movement[1])
        # CPU1 memory interactions handled separately
        return tb1_activity, self.cpu4.activity

    def generate_steering_command(self):
        # Default: random walk
        turn = np.random.normal(0, 0.3)
        if self.state == 'homing':
            # Use home vector
            x, y = self.cpu4.get_vector()
            desired_heading = np.arctan2(-y, -x)  # Home is at (0,0) in PI
            turn = self._angle_diff(desired_heading, self.heading) * 0.5
        elif self.state == 'memory_return' and self.current_target is not None:
            # Use memory vector
            mem_vec = self.cpu1.recall(self.current_target)
            x = np.sum(mem_vec * np.cos(self.cpu4.angles))
            y = np.sum(mem_vec * np.sin(self.cpu4.angles))
            desired_heading = np.arctan2(y, x)
            turn = self._angle_diff(desired_heading, self.heading) * 0.5
        # Clamp turn
        turn = np.clip(turn, -0.5, 0.5)
        return turn

    def _angle_diff(self, a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

    def store_vector_memory(self, reward_strength=1.0):
        self.cpu1.store(self.cpu4.activity, reward_strength)

    def recall_vector_memory(self, memory_id):
        return self.cpu1.recall(memory_id)

    def step(self):
        # Determine movement
        turn = self.generate_steering_command()
        self.heading = (self.heading + turn) % (2 * np.pi)
        dx = self.speed * self.dt * np.cos(self.heading)
        dy = self.speed * self.dt * np.sin(self.heading)
        # Boundary check
        new_pos = self.position + np.array([dx, dy])
        if np.any(new_pos < 0) or np.any(new_pos > self.env_size):
            self.heading = (self.heading + np.pi) % (2 * np.pi)  # Bounce
            dx, dy = -dx, -dy
        self.position += np.array([dx, dy])
        self.trajectory.append(self.position.copy())
        # Update neural activity
        self.update_neural_activity((dx, dy))
        self.time += self.dt
        # Handle behavioral transitions
        self._behavioral_logic()

    def _behavioral_logic(self):
        # Phase transitions and food/nest logic
        if self.phase == 1 and self.time > 30:
            # PHASE 2: Food discovery
            self.phase = 2
            food = self.position.copy()
            self.food_locations.append(food)
            self.store_vector_memory(reward_strength=1.0)
            self.state = 'homing'
        elif self.phase == 2 and np.linalg.norm(self.position - self.nest_location) < 3:
            # PHASE 3: Homing complete
            self.phase = 3
            self.state = 'memory_return'
            self.current_target = 0
        elif self.phase == 3 and np.linalg.norm(self.position - self.food_locations[0]) < 3:
            # PHASE 4: Return to food
            self.phase = 4
            self.state = 'exploration'
        elif self.phase == 4 and self.time > 60:
            # PHASE 5: Discover second food
            self.phase = 5
            food = self.position.copy()
            self.food_locations.append(food)
            self.store_vector_memory(reward_strength=1.0)
            self.state = 'homing'
        elif self.phase == 5 and np.linalg.norm(self.position - self.nest_location) < 3:
            # PHASE 6: Shortcut
            self.phase = 6
            self.state = 'shortcut'
            self.current_target = 1
        elif self.phase == 6 and np.linalg.norm(self.position - self.food_locations[1]) < 3:
            # PHASE 7: Route optimization
            self.phase = 7
            self.state = 'route_optimization'

    def run_simulation(self, steps=500):
        self.anim = FuncAnimation(self.fig, self._update_plot, frames=steps, interval=50, repeat=False)
        plt.show()

    def _update_plot(self, frame):
        self.step()
        self.axs[0,0].cla()
        self.axs[0,1].cla()
        self.axs[1,0].cla()
        self.axs[1,1].cla()
        # Spatial plot
        traj = np.array(self.trajectory)
        self.axs[0,0].plot(traj[:,0], traj[:,1], 'b-', alpha=0.5)
        self.axs[0,0].plot(self.position[0], self.position[1], 'ro', label='Agent')
        self.axs[0,0].plot(self.nest_location[0], self.nest_location[1], 'ks', label='Nest')
        for i, food in enumerate(self.food_locations):
            self.axs[0,0].plot(food[0], food[1], 'g*', label=f'Food {i+1}')
        self.axs[0,0].set_xlim(0, self.env_size)
        self.axs[0,0].set_ylim(0, self.env_size)
        self.axs[0,0].set_title('Spatial Trajectory')
        self.axs[0,0].legend()
        # TB1 activity
        self.axs[0,1].bar(np.arange(self.tb1.n), self.tb1.activity)
        self.axs[0,1].set_title('TB1 Compass Activity')
        # CPU4 activity
        self.axs[1,0].bar(np.arange(self.cpu4.n), self.cpu4.activity)
        self.axs[1,0].set_title('CPU4 Path Integrator')
        # CPU1 memory
        mem_strengths = self.cpu1.get_memory_strengths()
        self.axs[1,1].bar(np.arange(len(mem_strengths)), mem_strengths)
        self.axs[1,1].set_title('CPU1 Memory Strengths')
        # State
        self.fig.suptitle(f'Time: {self.time:.1f}s | State: {self.state} | Phase: {self.phase}')

# -----------------------------
# Example Usage
# -----------------------------

if __name__ == '__main__':
    navigator = CentralComplexNavigator()
    navigator.run_simulation(steps=600)

    # For parameter exploration, modify navigator attributes before run_simulation
    # For example:
    # navigator.cpu4.gain = 0.8
    # navigator.speed = 1.5
    # navigator.cpu1.n_memories = 6

"""
Instructions:
- Run this script to launch the simulation and visualization.
- Adjust parameters in the CentralComplexNavigator or neural classes for exploration.
- See docstrings and comments for biological explanations.
- Extend with additional features as described in complex.md for advanced research.
""" 