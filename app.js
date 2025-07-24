// Central Complex Navigation System - Implementation
class CentralComplexNavigator {
    constructor() {
        // Algorithm parameters from provided data
        this.params = {
            n_cols: 16,
            arena_size: 200,
            step_size: 1.0,
            max_steps: 60000,
            food_locations: [[150, 50], [50, 150]],
            nest_position: [100, 100],
            store_range: 4.0,
            path_decay: 0.9998,
            recalib_rate: 0.001,
            search_turns: [-30, 30],
            mem_reward: 1.0
        };

        // Agent state
        this.agent = {
            x: 100,
            y: 100,
            heading: 0,
            trail: [],
            state: 'exploration'
        };

        // Neural circuits
        this.tb1 = new Array(16).fill(0); // Ring attractor compass
        this.cpu4 = new Array(16).fill(0); // Path integration
        this.cpu1 = new Array(16).fill(0); // Memory comparison
        
        // Memory system
        this.memories = [];
        this.home_vector = { x: 0, y: 0 };
        
        // Simulation state
        this.running = false;
        this.step_count = 0;
        this.speed = 1.0;
        this.statistics = {
            food_found: 0,
            returns_home: 0,
            success_rate: 0
        };

        this.initializeCanvas();
        this.initializeControls();
        this.initializeNeuralDisplays();
        this.updateDisplays();
        this.startAnimation();
    }

    initializeCanvas() {
        this.canvas = document.getElementById('nav-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.scale = this.canvas.width / this.params.arena_size;
        
        // Canvas interaction
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleCanvasRightClick(e);
        });
    }

    initializeControls() {
        // Play/Pause button
        document.getElementById('play-pause-btn').addEventListener('click', () => {
            this.running = !this.running;
            document.getElementById('play-pause-text').textContent = this.running ? 'Pause' : 'Play';
        });

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSimulation();
        });

        // Speed slider
        document.getElementById('speed-slider').addEventListener('input', (e) => {
            this.speed = parseFloat(e.target.value);
            document.getElementById('speed-value').textContent = this.speed.toFixed(1) + 'x';
        });

        // Arena size
        document.getElementById('arena-size').addEventListener('input', (e) => {
            this.params.arena_size = parseInt(e.target.value);
            document.getElementById('arena-size-value').textContent = e.target.value;
            this.scale = this.canvas.width / this.params.arena_size;
        });

        // Food count
        document.getElementById('food-count').addEventListener('input', (e) => {
            const count = parseInt(e.target.value);
            document.getElementById('food-count-value').textContent = count;
            this.updateFoodLocations(count);
        });

        // Step size
        document.getElementById('step-size').addEventListener('input', (e) => {
            this.params.step_size = parseFloat(e.target.value);
            document.getElementById('step-size-value').textContent = e.target.value;
        });

        // Memory decay
        document.getElementById('memory-decay').addEventListener('input', (e) => {
            this.params.path_decay = parseFloat(e.target.value);
        });

        // Navigation modes
        document.querySelectorAll('input[name="nav-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.agent.state = e.target.value;
                    this.updateStateDisplay();
                }
            });
        });

        // Preset scenarios
        document.querySelectorAll('[data-preset]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.loadPreset(e.target.dataset.preset);
            });
        });

        // Canvas controls
        document.getElementById('clear-trail').addEventListener('click', () => {
            this.agent.trail = [];
        });

        document.getElementById('center-view').addEventListener('click', () => {
            this.centerAgent();
        });
    }

    initializeNeuralDisplays() {
        // Initialize TB1 compass display
        const tb1Display = document.getElementById('tb1-display');
        for (let i = 0; i < 16; i++) {
            const neuron = document.createElement('div');
            neuron.className = 'compass-neuron';
            neuron.style.transform = `rotate(${i * 22.5}deg) translateY(-60px)`;
            tb1Display.appendChild(neuron);
        }

        // Initialize CPU4 bar display
        const cpu4Display = document.getElementById('cpu4-display');
        for (let i = 0; i < 16; i++) {
            const bar = document.createElement('div');
            bar.className = 'cpu4-bar';
            cpu4Display.appendChild(bar);
        }

        // Initialize memory display
        this.updateMemoryDisplay();
    }

    // Core navigation algorithm
    step() {
        if (!this.running) return;

        // Update neural circuits
        this.updateTB1();
        this.updateCPU4();
        this.updateCPU1();

        // Behavioral control
        const action = this.selectAction();
        this.executeAction(action);

        // Check for food discovery
        this.checkFoodDiscovery();

        // Check for nest return
        this.checkNestReturn();

        // Update trail
        this.agent.trail.push({ x: this.agent.x, y: this.agent.y });
        if (this.agent.trail.length > 1000) {
            this.agent.trail.shift();
        }

        this.step_count++;
        this.updateDisplays();
    }

    updateTB1() {
        // Ring attractor compass - encodes current heading
        const heading_idx = Math.floor((this.agent.heading + 180) / 22.5) % 16;
        
        // Decay all neurons
        for (let i = 0; i < 16; i++) {
            this.tb1[i] *= 0.95;
        }
        
        // Activate current heading
        this.tb1[heading_idx] = Math.min(1.0, this.tb1[heading_idx] + 0.5);
        
        // Lateral inhibition
        for (let i = 0; i < 16; i++) {
            const left = (i - 1 + 16) % 16;
            const right = (i + 1) % 16;
            this.tb1[i] += 0.1 * (this.tb1[left] + this.tb1[right]);
        }
    }

    updateCPU4() {
        // Path integration - maintains home vector
        const dx = Math.cos(this.agent.heading * Math.PI / 180) * this.params.step_size;
        const dy = Math.sin(this.agent.heading * Math.PI / 180) * this.params.step_size;
        
        this.home_vector.x += dx;
        this.home_vector.y += dy;
        
        // Apply decay
        this.home_vector.x *= this.params.path_decay;
        this.home_vector.y *= this.params.path_decay;
        
        // Encode in sinusoidal pattern
        const home_distance = Math.sqrt(this.home_vector.x ** 2 + this.home_vector.y ** 2);
        const home_angle = Math.atan2(this.home_vector.y, this.home_vector.x);
        
        for (let i = 0; i < 16; i++) {
            const angle = i * Math.PI / 8;
            this.cpu4[i] = Math.max(0, Math.sin(angle - home_angle) * home_distance / 50);
        }
    }

    updateCPU1() {
        // Memory comparison - compares current location with stored memories
        for (let i = 0; i < 16; i++) {
            this.cpu1[i] = 0;
        }
        
        for (let mem of this.memories) {
            const dist = Math.sqrt((this.agent.x - mem.x) ** 2 + (this.agent.y - mem.y) ** 2);
            if (dist < this.params.store_range * 3) {
                const angle = Math.atan2(mem.y - this.agent.y, mem.x - this.agent.x);
                const idx = Math.floor((angle + Math.PI) / (Math.PI / 8)) % 16;
                this.cpu1[idx] = Math.max(this.cpu1[idx], mem.strength * (1 - dist / (this.params.store_range * 3)));
            }
        }
    }

    selectAction() {
        switch (this.agent.state) {
            case 'exploration':
                return this.exploreAction();
            case 'homing':
                return this.homeAction();
            case 'food-return':
                return this.foodReturnAction();
            case 'manual':
                return this.manualAction();
            default:
                return this.exploreAction();
        }
    }

    exploreAction() {
        // Random walk with slight bias toward areas with high memory activity
        let turn = (Math.random() - 0.5) * 60; // Random turn
        
        // Add memory bias
        const max_memory = Math.max(...this.cpu1);
        if (max_memory > 0.1) {
            const memory_idx = this.cpu1.indexOf(max_memory);
            const memory_angle = memory_idx * 22.5 - 180;
            const angle_diff = ((memory_angle - this.agent.heading + 540) % 360) - 180;
            turn += angle_diff * 0.3; // Bias toward memory
        }
        
        return { type: 'turn', amount: turn };
    }

    homeAction() {
        // Use path integration to head home
        const home_angle = Math.atan2(-this.home_vector.y, -this.home_vector.x) * 180 / Math.PI;
        const angle_diff = ((home_angle - this.agent.heading + 540) % 360) - 180;
        return { type: 'turn', amount: angle_diff * 0.5 };
    }

    foodReturnAction() {
        // Navigate to stored food location
        if (this.memories.length === 0) {
            return this.exploreAction();
        }
        
        // Find strongest memory
        const best_memory = this.memories.reduce((best, mem) => 
            mem.strength > best.strength ? mem : best, this.memories[0]);
        
        const target_angle = Math.atan2(best_memory.y - this.agent.y, best_memory.x - this.agent.x) * 180 / Math.PI;
        const angle_diff = ((target_angle - this.agent.heading + 540) % 360) - 180;
        
        return { type: 'turn', amount: angle_diff * 0.7 };
    }

    manualAction() {
        // Manual control would be implemented with keyboard input
        return { type: 'turn', amount: 0 };
    }

    executeAction(action) {
        if (action.type === 'turn') {
            this.agent.heading += action.amount;
            this.agent.heading = ((this.agent.heading % 360) + 360) % 360;
        }
        
        // Move forward
        const dx = Math.cos(this.agent.heading * Math.PI / 180) * this.params.step_size;
        const dy = Math.sin(this.agent.heading * Math.PI / 180) * this.params.step_size;
        
        this.agent.x += dx;
        this.agent.y += dy;
        
        // Boundary checking
        this.agent.x = Math.max(10, Math.min(this.params.arena_size - 10, this.agent.x));
        this.agent.y = Math.max(10, Math.min(this.params.arena_size - 10, this.agent.y));
    }

    checkFoodDiscovery() {
        for (let food of this.params.food_locations) {
            const dist = Math.sqrt((this.agent.x - food[0]) ** 2 + (this.agent.y - food[1]) ** 2);
            if (dist < this.params.store_range) {
                // Store memory
                const existing = this.memories.find(m => 
                    Math.sqrt((m.x - food[0]) ** 2 + (m.y - food[1]) ** 2) < this.params.store_range);
                
                if (!existing) {
                    this.memories.push({
                        x: food[0],
                        y: food[1],
                        strength: this.params.mem_reward,
                        discovered_at: this.step_count
                    });
                    this.statistics.food_found++;
                    this.agent.state = 'homing';
                    this.updateStateDisplay();
                }
                break;
            }
        }
    }

    checkNestReturn() {
        const nest_dist = Math.sqrt(
            (this.agent.x - this.params.nest_position[0]) ** 2 + 
            (this.agent.y - this.params.nest_position[1]) ** 2
        );
        
        if (nest_dist < this.params.store_range) {
            if (this.agent.state === 'homing') {
                // Recalibrate path integration
                this.home_vector.x *= this.params.recalib_rate;
                this.home_vector.y *= this.params.recalib_rate;
                
                this.statistics.returns_home++;
                this.agent.state = 'exploration';
                this.updateStateDisplay();
            }
        }
    }

    // UI Update methods
    updateDisplays() {
        this.drawCanvas();
        this.updateNeuralDisplays();
        this.updateStatistics();
    }

    drawCanvas() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw arena boundary
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw trail
        if (this.agent.trail.length > 1) {
            ctx.strokeStyle = 'rgba(45, 166, 178, 0.6)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(this.agent.trail[0].x * this.scale, this.agent.trail[0].y * this.scale);
            for (let i = 1; i < this.agent.trail.length; i++) {
                ctx.lineTo(this.agent.trail[i].x * this.scale, this.agent.trail[i].y * this.scale);
            }
            ctx.stroke();
        }
        
        // Draw nest
        const nest = this.params.nest_position;
        ctx.fillStyle = '#33808d';
        ctx.beginPath();
        ctx.arc(nest[0] * this.scale, nest[1] * this.scale, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw food sources
        ctx.fillStyle = '#27ae60';
        for (let food of this.params.food_locations) {
            ctx.beginPath();
            ctx.arc(food[0] * this.scale, food[1] * this.scale, 6, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // Draw agent
        ctx.fillStyle = '#e67e22';
        ctx.beginPath();
        ctx.arc(this.agent.x * this.scale, this.agent.y * this.scale, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw heading arrow
        ctx.strokeStyle = '#c0392b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.agent.x * this.scale, this.agent.y * this.scale);
        const arrow_x = this.agent.x * this.scale + Math.cos(this.agent.heading * Math.PI / 180) * 15;
        const arrow_y = this.agent.y * this.scale + Math.sin(this.agent.heading * Math.PI / 180) * 15;
        ctx.lineTo(arrow_x, arrow_y);
        ctx.stroke();
    }

    updateNeuralDisplays() {
        // Update TB1 compass display
        const compassNeurons = document.querySelectorAll('.compass-neuron');
        compassNeurons.forEach((neuron, i) => {
            const activity = this.tb1[i];
            const intensity = Math.floor(activity * 255);
            neuron.style.backgroundColor = `rgb(${intensity}, ${Math.floor(intensity * 0.7)}, ${Math.floor(intensity * 0.3)})`;
        });
        
        // Update heading indicator
        document.getElementById('heading-indicator').style.transform = `rotate(${this.agent.heading}deg)`;
        
        // Update CPU4 bars
        const cpu4Bars = document.querySelectorAll('.cpu4-bar');
        cpu4Bars.forEach((bar, i) => {
            const activity = this.cpu4[i];
            bar.style.height = `${activity * 80}px`;
            bar.style.backgroundColor = activity > 0.1 ? '#f39c12' : '#ecf0f1';
        });
        
        // Update home vector info
        const home_distance = Math.sqrt(this.home_vector.x ** 2 + this.home_vector.y ** 2);
        const home_direction = Math.atan2(this.home_vector.y, this.home_vector.x) * 180 / Math.PI;
        document.getElementById('home-distance').textContent = home_distance.toFixed(1);
        document.getElementById('home-direction').textContent = home_direction.toFixed(0) + '°';
        
        // Update memory display
        this.updateMemoryDisplay();
    }

    updateMemoryDisplay() {
        const memoryDisplay = document.getElementById('cpu1-display');
        memoryDisplay.innerHTML = '';
        
        this.memories.forEach((memory, i) => {
            const memoryBar = document.createElement('div');
            memoryBar.className = 'memory-bar';
            
            const label = document.createElement('span');
            label.className = 'memory-label';
            label.textContent = `Food ${i + 1}`;
            
            const strength = document.createElement('div');
            strength.className = 'memory-strength';
            
            const fill = document.createElement('div');
            fill.className = 'memory-strength-fill';
            fill.style.width = `${memory.strength * 100}%`;
            
            strength.appendChild(fill);
            memoryBar.appendChild(label);
            memoryBar.appendChild(strength);
            memoryDisplay.appendChild(memoryBar);
        });
        
        document.getElementById('active-memories').textContent = this.memories.length;
    }

    updateStatistics() {
        document.getElementById('step-counter').textContent = this.step_count;
        document.getElementById('food-found').textContent = this.statistics.food_found;
        document.getElementById('returns-home').textContent = this.statistics.returns_home;
        
        const success_rate = this.statistics.food_found > 0 ? 
            (this.statistics.returns_home / this.statistics.food_found * 100).toFixed(1) : 0;
        document.getElementById('success-rate').textContent = success_rate + '%';
        this.statistics.success_rate = success_rate;
    }

    updateStateDisplay() {
        const stateElement = document.getElementById('current-state');
        stateElement.textContent = this.agent.state.charAt(0).toUpperCase() + this.agent.state.slice(1);
        
        // Update status class
        stateElement.className = 'status';
        switch (this.agent.state) {
            case 'exploration':
                stateElement.classList.add('status--info');
                break;
            case 'homing':
                stateElement.classList.add('status--warning');
                break;
            case 'food-return':
                stateElement.classList.add('status--success');
                break;
            default:
                stateElement.classList.add('status--info');
        }
    }

    // Event handlers
    handleCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale;
        const y = (e.clientY - rect.top) / this.scale;
        
        // Add food source
        this.params.food_locations.push([x, y]);
        document.getElementById('food-count').value = this.params.food_locations.length;
        document.getElementById('food-count-value').textContent = this.params.food_locations.length;
    }

    handleCanvasRightClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale;
        const y = (e.clientY - rect.top) / this.scale;
        
        // Move nest
        this.params.nest_position = [x, y];
    }

    // Utility methods
    resetSimulation() {
        this.agent.x = this.params.nest_position[0];
        this.agent.y = this.params.nest_position[1];
        this.agent.heading = Math.random() * 360;
        this.agent.trail = [];
        this.agent.state = 'exploration';
        
        this.home_vector = { x: 0, y: 0 };
        this.memories = [];
        this.step_count = 0;
        
        this.statistics = {
            food_found: 0,
            returns_home: 0,
            success_rate: 0
        };
        
        this.running = false;
        document.getElementById('play-pause-text').textContent = 'Play';
        
        this.updateDisplays();
        this.updateStateDisplay();
    }

    updateFoodLocations(count) {
        while (this.params.food_locations.length < count) {
            const x = Math.random() * (this.params.arena_size - 40) + 20;
            const y = Math.random() * (this.params.arena_size - 40) + 20;
            this.params.food_locations.push([x, y]);
        }
        while (this.params.food_locations.length > count) {
            this.params.food_locations.pop();
        }
    }

    loadPreset(preset) {
        this.resetSimulation();
        
        switch (preset) {
            case 'basic':
                this.params.food_locations = [[150, 50]];
                break;
            case 'multi':
                this.params.food_locations = [[150, 50], [50, 150], [150, 150]];
                break;
            case 'shortcuts':
                this.params.food_locations = [[170, 30], [30, 170]];
                break;
        }
        
        document.getElementById('food-count').value = this.params.food_locations.length;
        document.getElementById('food-count-value').textContent = this.params.food_locations.length;
    }

    centerAgent() {
        this.agent.x = this.params.arena_size / 2;
        this.agent.y = this.params.arena_size / 2;
    }

    startAnimation() {
        const animate = () => {
            for (let i = 0; i < this.speed; i++) {
                this.step();
            }
            requestAnimationFrame(animate);
        };
        animate();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.navigator = new CentralComplexNavigator();
});