"""
Central Complex Navigation System (Exploration Enhanced)
======================================================

This version adds a coverage map and exploration bias, so the agent systematically explores the environment and avoids repeatedly visiting the same areas. During exploration, the agent is biased toward less-visited regions, while retaining some randomness for biological plausibility.

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# -----------------------------
# Neural Circuit Classes (same as before)
# -----------------------------

class TB1Compass:
    def __init__(self, noise_std=0.05):
        self.n = 8
        self.angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.activity = np.zeros(self.n)
        self.noise_std = noise_std
    def update(self, heading_rad):
        self.activity = np.cos(self.angles - heading_rad)
        self.activity += np.random.normal(0, self.noise_std, self.n)
        self.activity = np.clip(self.activity, -1, 1)
        return self.activity

class CPU4Integrator:
    def __init__(self, noise_std=0.05, gain=1.0, decay=0.0):
        self.n = 16
        self.angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.activity = np.zeros(self.n)
        self.noise_std = noise_std
        self.gain = gain
        self.decay = decay
    def integrate(self, delta_x, delta_y):
        movement_angle = np.arctan2(delta_y, delta_x)
        movement_mag = np.hypot(delta_x, delta_y)
        self.activity *= (1 - self.decay)
        self.activity += self.gain * movement_mag * np.cos(self.angles - movement_angle)
        self.activity += np.random.normal(0, self.noise_std, self.n)
        return self.activity
    def get_vector(self):
        x = np.sum(self.activity * np.cos(self.angles))
        y = np.sum(self.activity * np.sin(self.angles))
        return x, y
    def reset(self):
        self.activity = np.zeros(self.n)

class CPU1Memory:
    def __init__(self, n_memories=4, interference=0.0):
        self.n = 16
        self.n_memories = n_memories
        self.memories = []
        self.interference = interference
    def store(self, cpu4_activity, reward_strength=1.0):
        if len(self.memories) >= self.n_memories:
            self.memories.pop(0)
        noisy_activity = cpu4_activity.copy() + np.random.normal(0, self.interference, self.n)
        self.memories.append((noisy_activity, reward_strength))
    def recall(self, memory_id=0):
        if 0 <= memory_id < len(self.memories):
            return self.memories[memory_id][0]
        else:
            return np.zeros(self.n)
    def get_memory_strengths(self):
        return [s for _, s in self.memories]
    def consolidate(self, memory_id):
        if 0 <= memory_id < len(self.memories):
            activity, strength = self.memories[memory_id]
            self.memories[memory_id] = (activity, min(strength + 0.1, 2.0))

# -----------------------------
# Central Complex Navigator (with coverage map)
# -----------------------------

class CentralComplexNavigator:
    def __init__(self, env_size=100, dt=0.2, cpu4_gain=1.0, cpu4_decay=0.0, cpu1_interference=0.0, coverage_resolution=2):
        self.env_size = env_size
        self.dt = dt
        self.position = np.array([env_size/2, env_size/2], dtype=float)
        self.heading = random.uniform(0, 2*np.pi)
        self.speed = 1.0
        self.state = 'exploration'
        self.trajectory = [self.position.copy()]
        self.food_locations = []
        self.nest_location = np.array([env_size/2, env_size/2])
        self.current_target = None
        self.time = 0.0
        self.phase = 1
        self.tb1 = TB1Compass()
        self.cpu4 = CPU4Integrator(gain=cpu4_gain, decay=cpu4_decay)
        self.cpu1 = CPU1Memory(interference=cpu1_interference)
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 9))
        plt.tight_layout()
        self.anim = None
        self.performance_metrics = []
        # Coverage map
        self.coverage_resolution = coverage_resolution
        self.coverage_shape = (env_size // coverage_resolution, env_size // coverage_resolution)
        self.coverage_map = np.zeros(self.coverage_shape, dtype=int)

    def update_neural_activity(self, movement, sensory_input=None):
        tb1_activity = self.tb1.update(self.heading)
        self.cpu4.integrate(movement[0], movement[1])
        return tb1_activity, self.cpu4.activity

    def _update_coverage(self):
        x_idx = int(self.position[0] // self.coverage_resolution)
        y_idx = int(self.position[1] // self.coverage_resolution)
        if 0 <= x_idx < self.coverage_shape[0] and 0 <= y_idx < self.coverage_shape[1]:
            self.coverage_map[x_idx, y_idx] += 1

    def generate_steering_command(self):
        if self.state == 'exploration':
            # Bias toward less-visited areas
            look_angles = np.linspace(-np.pi/2, np.pi/2, 9)  # 9 directions
            scores = []
            for da in look_angles:
                test_heading = (self.heading + da) % (2*np.pi)
                test_x = self.position[0] + np.cos(test_heading) * self.speed * self.dt * 5
                test_y = self.position[1] + np.sin(test_heading) * self.speed * self.dt * 5
                x_idx = int(test_x // self.coverage_resolution)
                y_idx = int(test_y // self.coverage_resolution)
                if 0 <= x_idx < self.coverage_shape[0] and 0 <= y_idx < self.coverage_shape[1]:
                    score = -self.coverage_map[x_idx, y_idx]  # Prefer less-visited
                else:
                    score = -1000  # Penalize out-of-bounds
                scores.append(score)
            best_idx = np.argmax(scores)
            best_angle = look_angles[best_idx]
            # Add some randomness
            turn = best_angle + np.random.normal(0, 0.2)
            turn = np.clip(turn, -0.5, 0.5)
            return turn
        # All other states as before
        turn = np.random.normal(0, 0.3)
        if self.state == 'homing':
            x, y = self.cpu4.get_vector()
            desired_heading = np.arctan2(-y, -x)
            turn = self._angle_diff(desired_heading, self.heading) * 0.5
        elif self.state == 'memory_return' and self.current_target is not None:
            mem_vec = self.cpu1.recall(self.current_target)
            x = np.sum(mem_vec * np.cos(self.cpu4.angles))
            y = np.sum(mem_vec * np.sin(self.cpu4.angles))
            desired_heading = np.arctan2(y, x)
            turn = self._angle_diff(desired_heading, self.heading) * 0.5
        elif self.state == 'shortcut' and len(self.food_locations) > 1:
            vec = self.food_locations[1] - self.food_locations[0]
            desired_heading = np.arctan2(vec[1], vec[0])
            turn = self._angle_diff(desired_heading, self.heading) * 0.5
        elif self.state == 'route_optimization' and len(self.food_locations) > 1:
            dists = [np.linalg.norm(self.position - f) for f in self.food_locations]
            idx = np.argmin(dists)
            vec = self.food_locations[idx] - self.position
            desired_heading = np.arctan2(vec[1], vec[0])
            turn = self._angle_diff(desired_heading, self.heading) * 0.5
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
        turn = self.generate_steering_command()
        self.heading = (self.heading + turn) % (2 * np.pi)
        dx = self.speed * self.dt * np.cos(self.heading)
        dy = self.speed * self.dt * np.sin(self.heading)
        new_pos = self.position + np.array([dx, dy])
        if np.any(new_pos < 0) or np.any(new_pos > self.env_size):
            self.heading = (self.heading + np.pi) % (2 * np.pi)
            dx, dy = -dx, -dy
        self.position += np.array([dx, dy])
        self.trajectory.append(self.position.copy())
        self.update_neural_activity((dx, dy))
        self._update_coverage()
        self.time += self.dt
        self._behavioral_logic()
        self._update_performance_metrics()

    def _behavioral_logic(self):
        if self.phase == 1 and self.time > 30:
            self.phase = 2
            food = self.position.copy()
            self.food_locations.append(food)
            self.store_vector_memory(reward_strength=1.0)
            self.state = 'homing'
        elif self.phase == 2 and np.linalg.norm(self.position - self.nest_location) < 3:
            self.phase = 3
            self.state = 'memory_return'
            self.current_target = 0
        elif self.phase == 3 and np.linalg.norm(self.position - self.food_locations[0]) < 3:
            self.phase = 4
            self.state = 'exploration'
        elif self.phase == 4 and self.time > 60:
            self.phase = 5
            food = self.position.copy()
            self.food_locations.append(food)
            self.store_vector_memory(reward_strength=1.0)
            self.state = 'homing'
        elif self.phase == 5 and np.linalg.norm(self.position - self.nest_location) < 3:
            self.phase = 6
            self.state = 'shortcut'
            self.current_target = 1
        elif self.phase == 6 and np.linalg.norm(self.position - self.food_locations[1]) < 3:
            self.phase = 7
            self.state = 'route_optimization'

    def _update_performance_metrics(self):
        if self.food_locations:
            dist_to_nest = np.linalg.norm(self.position - self.nest_location)
            dist_to_food = np.linalg.norm(self.position - self.food_locations[-1])
            self.performance_metrics.append((self.time, dist_to_nest, dist_to_food))

    def run_simulation(self, steps=700):
        self.anim = FuncAnimation(self.fig, self._update_plot, frames=steps, interval=50, repeat=False)
        plt.show()

    def _update_plot(self, frame):
        self.step()
        for ax in self.axs.flat:
            ax.cla()
        traj = np.array(self.trajectory)
        self.axs[0,0].plot(traj[:,0], traj[:,1], 'b-', alpha=0.5, label='Trajectory')
        self.axs[0,0].plot(self.position[0], self.position[1], 'ro', label='Agent')
        self.axs[0,0].plot(self.nest_location[0], self.nest_location[1], 'ks', label='Nest')
        for i, food in enumerate(self.food_locations):
            self.axs[0,0].plot(food[0], food[1], 'g*', markersize=12, label=f'Food {i+1}')
        self.axs[0,0].set_xlim(0, self.env_size)
        self.axs[0,0].set_ylim(0, self.env_size)
        self.axs[0,0].set_title('Spatial Trajectory')
        self.axs[0,0].legend(loc='upper right', fontsize=8)
        # TB1 activity
        self.axs[0,1].bar(np.arange(self.tb1.n), self.tb1.activity, color='orange')
        self.axs[0,1].set_title('TB1 Compass Activity')
        self.axs[0,1].set_xlabel('Neuron')
        self.axs[0,1].set_ylabel('Activity')
        # CPU4 activity
        self.axs[1,0].bar(np.arange(self.cpu4.n), self.cpu4.activity, color='purple')
        self.axs[1,0].set_title('CPU4 Path Integrator')
        self.axs[1,0].set_xlabel('Neuron')
        self.axs[1,0].set_ylabel('Activity')
        # CPU1 memory
        mem_strengths = self.cpu1.get_memory_strengths()
        self.axs[1,1].bar(np.arange(len(mem_strengths)), mem_strengths, color='green')
        self.axs[1,1].set_title('CPU1 Memory Strengths')
        self.axs[1,1].set_xlabel('Memory Slot')
        self.axs[1,1].set_ylabel('Strength')
        # Coverage map
        self.axs[1,1].imshow(self.coverage_map.T, origin='lower', cmap='Blues', alpha=0.3, extent=[0, self.env_size, 0, self.env_size], aspect='auto')
        # State
        self.fig.suptitle(f'Time: {self.time:.1f}s | State: {self.state} | Phase: {self.phase}')

# -----------------------------
# Example Usage
# -----------------------------

if __name__ == '__main__':
    navigator = CentralComplexNavigator(cpu4_gain=1.0, cpu4_decay=0.01, cpu1_interference=0.02, coverage_resolution=2)
    navigator.run_simulation(steps=900)

    # For parameter exploration, modify navigator attributes before run_simulation
    # For example:
    # navigator.cpu4.gain = 0.8
    # navigator.speed = 1.5
    # navigator.cpu1.n_memories = 6
    # navigator.cpu1.consolidate(0)

"""
Instructions:
- Run this script to launch the exploration-enhanced simulation and visualization.
- The agent will systematically explore the environment during the exploration phase.
- The coverage map is shown as a blue heatmap overlay.
- Adjust parameters in the CentralComplexNavigator or neural classes for exploration.
- See docstrings and comments for biological explanations and extension points.
- Extend with additional features as described in complex.md for advanced research.
""" 