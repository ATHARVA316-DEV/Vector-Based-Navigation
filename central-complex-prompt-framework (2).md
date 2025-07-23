# Central Complex Navigation System: AI Prompt Framework

## Overview
This document provides a comprehensive prompt engineering framework for instructing AI systems to create the Central Complex Navigation System based on the Le Moël et al. (2019) research on insect vector-based navigation.

## Core Prompt Architecture

### Part 1: System Initialization & Context Setting

```
ROLE: You are an expert computational neuroscientist and bio-inspired robotics engineer specializing in insect navigation systems and neural circuit modeling.

CONTEXT: You will create a complete implementation of the Central Complex vector-based navigation system based on the research paper "The Central Complex as a Potential Substrate for Vector Based Navigation" by Le Moël et al. (2019). This system models how insects like bees and ants perform sophisticated navigation using minimal neural circuits.

OBJECTIVE: Generate a fully functional navigation system that includes:
1. Biologically accurate neural circuit models
2. Path integration mechanics
3. Vector memory storage and recall
4. Visual simulation capabilities
5. Real-time neural activity visualization

CONSTRAINTS:
- Must be biologically plausible based on insect neuroanatomy
- Should demonstrate all behaviors described in the research paper
- Code must be modular and well-documented
- Include both spatial and neural visualizations
- Implement realistic insect movement dynamics
```

### Part 2: Technical Specifications Module

```
TECHNICAL REQUIREMENTS:

NEURAL ARCHITECTURE:
- TB1 neurons (8 cells): Compass system with ring attractor dynamics
- CPU4 neurons (16 cells): Path integration system encoding home vector
- CPU1 neurons (16 cells): Memory system for stored food locations
- Implement mutual inhibition and cross-connections between neural populations

MATHEMATICAL MODELS:
- Use sinusoidal activity patterns for distributed encoding
- Implement phase-based directional encoding
- Model synaptic plasticity for memory storage
- Include noise and biological variability

BEHAVIORAL CAPABILITIES:
1. Random exploration with realistic movement patterns
2. Path integration-based homing
3. Vector memory storage triggered by reward
4. Memory-guided navigation to stored locations
5. Novel shortcut computation between any two known locations
6. Multi-location route optimization

VISUALIZATION REQUIREMENTS:
- Real-time spatial movement with trajectory tracking
- Live neural activity displays (bar charts for each neuron type)
- Clear indication of current behavioral state
- Memory storage visualization
- Movement direction and path integration vector display
```

### Part 3: Implementation Strategy Module

```
IMPLEMENTATION APPROACH:

STEP-BY-STEP BREAKDOWN:
1. Create base neural circuit classes with proper mathematical foundations
2. Implement core navigation behaviors as separate, testable modules
3. Build visualization system with real-time updates
4. Integrate all components into main simulation loop
5. Add parameter controls and interactive features

CODING STANDARDS:
- Use object-oriented design with clear class hierarchies
- Include comprehensive docstrings explaining biological basis
- Implement proper error handling and edge cases
- Create modular functions that can be easily modified
- Use numpy for efficient numerical computations

BIOLOGICAL ACCURACY:
- Base all parameters on experimental measurements when available
- Include realistic time constants and neural dynamics
- Model both electrical and chemical aspects of neural communication
- Implement proper scaling between neural activity and behavior

OUTPUT FORMAT:
- Complete Python implementation with all necessary imports
- Detailed comments explaining each biological component
- Example usage and parameter settings
- Instructions for running and modifying the simulation
```

### Part 4: Behavioral Demonstration Module

```
DEMONSTRATION SEQUENCE:

The system must showcase these specific navigation behaviors in order:

PHASE 1: EXPLORATION (30 seconds)
- Random walk behavior with realistic turning patterns
- Neural activity showing compass and path integration updates
- Gradual accumulation of home vector

PHASE 2: FOOD DISCOVERY (Food Location A)
- Detection and approach to first food source
- Vector memory storage triggered by reward signal
- Clear visualization of memory neuron activation

PHASE 3: HOMING BEHAVIOR
- Direct navigation back to nest using path integration
- Neural activity showing home vector guidance
- Realistic search behavior upon reaching home area

PHASE 4: MEMORY-GUIDED RETURN
- Navigation from nest back to stored food location A
- Demonstration of vector memory recall and guidance
- Comparison between stored and actual locations

PHASE 5: SECOND LOCATION DISCOVERY (Food Location B)
- Discovery and storage of second food location
- Multiple memory storage demonstration

PHASE 6: SHORTCUT BEHAVIOR
- Novel direct navigation from food A to food B
- Vector arithmetic computation in neural circuits
- No prior experience with this specific route

PHASE 7: ROUTE OPTIMIZATION
- Efficient visiting sequence for multiple locations
- Nearest-neighbor route selection
- Trapline formation demonstration
```

### Part 5: Advanced Features Module

```
ADVANCED IMPLEMENTATIONS:

NEURAL PLASTICITY:
- Implement memory consolidation over multiple visits
- Show how repeated use strengthens vector memories
- Model memory interference and capacity limitations

ERROR CORRECTION:
- Memory recalibration based on navigation errors
- Adaptive adjustment of stored vector weights
- Realistic handling of cumulative path integration drift

ENVIRONMENTAL INTERACTIONS:
- Boundary collision detection and avoidance
- Landmark influence on navigation (optional extension)
- Wind drift and environmental noise effects

PARAMETER EXPLORATION:
- Interactive controls for key biological parameters
- Real-time adjustment of neural gain and time constants
- Exploration of failure modes and edge cases

VALIDATION METRICS:
- Success rate measurements for different navigation tasks
- Trajectory analysis (straightness, efficiency)
- Neural activity correlation with behavioral states
- Comparison with experimental data from literature
```

### Part 6: Code Structure Template

```
EXPECTED CODE ORGANIZATION:

class CentralComplexNavigator:
    def __init__(self):
        # Initialize neural circuits
        # Set up visualization components
        # Configure simulation parameters
    
    def update_neural_activity(self, movement, sensory_input):
        # Update TB1 compass neurons
        # Update CPU4 path integration
        # Process CPU1 memory interactions
        # Return neural states
    
    def generate_steering_command(self):
        # Compare current vs desired heading
        # Generate turn signal
        # Apply movement dynamics
        # Return movement vector
    
    def store_vector_memory(self, reward_strength):
        # Capture current CPU4 state
        # Store as synaptic weights
        # Update memory visualization
    
    def recall_vector_memory(self, memory_id):
        # Activate stored memory pattern
        # Modulate path integration output
        # Guide navigation toward stored location
    
    def run_simulation(self):
        # Main simulation loop
        # Handle behavioral state transitions
        # Update visualizations
        # Process user interactions

VISUALIZATION COMPONENTS:
- Spatial plot with insect position and trajectory
- Neural activity bar charts (TB1, CPU4, CPU1)
- Vector displays (current heading, home vector, memory vectors)
- Behavioral state indicator
- Performance metrics display
```

### Part 7: Quality Assurance Module

```
TESTING AND VALIDATION:

FUNCTIONALITY TESTS:
- Verify each neural circuit operates within biological ranges
- Test all navigation behaviors independently
- Validate vector arithmetic computations
- Check visualization accuracy and responsiveness

BIOLOGICAL PLAUSIBILITY:
- Compare neural activity patterns with experimental data
- Verify realistic movement dynamics and speeds
- Test parameter sensitivity and robustness
- Ensure proper scaling between neural and behavioral levels

PERFORMANCE BENCHMARKS:
- Measure navigation success rates (should exceed 90% for trained routes)
- Evaluate trajectory efficiency and straightness
- Test system performance with multiple concurrent memories
- Validate error correction and adaptation capabilities

EDGE CASE HANDLING:
- Test behavior at environment boundaries
- Handle memory capacity limitations
- Process conflicting or corrupted memories
- Manage extreme parameter values gracefully

DOCUMENTATION REQUIREMENTS:
- Clear explanation of biological basis for each component
- Parameter descriptions with biological justification
- Usage examples and troubleshooting guide
- Extension points for future development
```

## Usage Instructions

To use this framework:

1. **Initialize the conversation** with the system initialization prompt
2. **Provide technical specifications** to ensure biological accuracy
3. **Request step-by-step implementation** following the behavioral modules
4. **Iterate and refine** specific components as needed
5. **Test and validate** using the quality assurance criteria

## Expected Output

The AI system should generate:
- Complete Python implementation (500-800 lines)
- Real-time visualization system
- Detailed biological explanations
- Interactive parameter controls
- Comprehensive testing examples

This framework ensures the AI creates a scientifically accurate, fully functional, and visually compelling implementation of the central complex navigation system.