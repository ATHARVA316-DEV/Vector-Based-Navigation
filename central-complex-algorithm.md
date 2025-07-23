# Central Complex Vector-Based Navigation Algorithm

## Overview
This algorithm implements the insect central complex neural model for vector-based navigation as described by Le Moël et al. (2019). The system enables path integration, vector memory storage/recall, shortcutting between locations, and efficient multi-location routing.

## Core Components

### 1. Neural Layer Structure

#### Layer 1: Sensory Input
- **Speed Input (TN2 neurons)**: Process optic flow information
- **Directional Input (TL neurons)**: Process compass information from polarized light

#### Layer 2: Signal Inversion (CL1 neurons)
- Invert directional signals from Layer 1

#### Layer 3: Compass Encoding (TB1 neurons) 
- 8 neurons representing azimuthal directions
- Implement ring attractor with mutual inhibition
- Maintain stable sinusoidal activity pattern

#### Layer 4: Path Integration (CPU4 neurons)
- 16 integrator neurons accumulating movement vectors
- Encode home vector as distributed sinusoidal pattern
- Phase indicates direction, amplitude indicates distance

#### Layer 5: Steering Output (CPU1 neurons)
- Compare current heading to desired heading
- Generate left/right turn signals

### Vector Memory System
- **Vector-memory neurons**: Store snapshots of CPU4 activity
- **Reinforcer neuron**: Trigger memory storage at food locations
- **Recalibrator neuron**: Update memories based on navigation errors

## Main Algorithm

### Initialize System
```
Initialize all neural layers with baseline activity (0.5)
Set accumulation rate (acc = 0.0025)
Set decay rate (decay = 0.1) 
Set noise parameters (σ = 0.1)
Initialize empty vector memory bank
Set current position to nest (0, 0)
```

### Core Navigation Loop

#### Step 1: Sensory Processing
```
For each time step t:
  
  // Process speed input
  speed_left = cos(heading + π/4) * velocity
  speed_right = cos(heading - π/4) * velocity
  TN2_L = max(0, min(1, speed_left))
  TN2_R = max(0, min(1, speed_right))
  
  // Process compass input  
  For each TL neuron i with preferred direction α_i:
    TL[i] = cos(α_i - current_heading)
  
  // Invert signals
  For each CL1 neuron i:
    CL1[i] = -TL[i]
```

#### Step 2: Compass Ring Attractor (TB1)
```
For each TB1 neuron i:
  excitation = sum of corresponding CL1 inputs
  inhibition = sum over j of: d * (cos(α_i - α_j) - 1)² * TB1[j]
  TB1[i] = sigmoid(excitation + inhibition)
```

#### Step 3: Path Integration (CPU4)
```
For each CPU4 neuron i:
  // Accumulate movement opposite to heading direction
  CPU4_input[i] = CPU4_input[i] + acc * (TN2 - TB1[i] - decay)
  CPU4_input[i] = clip(CPU4_input[i], 0, 1)
  CPU4[i] = sigmoid(CPU4_input[i]) + noise
```

#### Step 4: Vector Memory Operations

##### Memory Storage (at food location)
```
If at_food_location AND reinforcer_active:
  For each synapse i in new vector memory:
    memory_weights[i] = -sigmoid(CPU4_input[i])  // Store negated values
  Add memory to memory_bank
```

##### Memory Recall (when navigating to stored location)
```
If recalling_memory k:
  For each CPU4 neuron i:
    modified_output[i] = CPU4[i] + memory_weights[k][i]
    modified_output[i] = clip(modified_output[i], 0, 1)
```

##### Memory Recalibration (at nest)
```
If at_nest AND recalibrator_active:
  error_vector = CPU4_output  // Remaining activity indicates error
  For active memory k:
    For each synapse i:
      memory_weights[k][i] += recalibration_rate * (0.5 - error_vector[i])
```

#### Step 5: Steering Control (CPU1)
```
// Determine active input based on navigation mode
If exploring:
  steering_input = TB1  // Only compass input
  
Else if homing:
  steering_input = TB1 + CPU4  // Compass + home vector
  
Else if using_vector_memory:
  steering_input = TB1 + modified_CPU4_output  // With memory influence

// Generate steering signals with offset connections
For each CPU1 neuron i:
  CPU1_L[i] = sigmoid(steering_input[(i-1) mod 8])  // Left offset
  CPU1_R[i] = sigmoid(steering_input[(i+1) mod 8])  // Right offset

// Calculate turn angle
left_activity = sum(CPU1_L)
right_activity = sum(CPU1_R)
turn_angle = 0.5 * (left_activity - right_activity)
```

### Multi-Location Navigation Algorithm

#### Memory Selection for Traplines
```
Function select_next_memory():
  min_score = infinity
  selected_memory = null
  
  For each available memory k:
    // Calculate distance score
    score = 0
    For each CPU4 neuron i:
      modified_value = CPU4[i] + memory_weights[k][i]
      modified_value = clip(modified_value, 0, 1)
      score += modified_value
    
    If score < min_score:
      min_score = score
      selected_memory = k
  
  Return selected_memory
```

#### Multi-Location Route Execution
```
Function execute_trapline():
  available_memories = copy(all_stored_memories)
  
  While available_memories not empty:
    current_memory = select_next_memory(available_memories)
    navigate_using_memory(current_memory)
    
    If reached_target():
      remove current_memory from available_memories
      mark_location_as_visited()
    
    If timeout_reached():
      break
  
  // Return home using path integration
  navigate_home()
```

### Shortcutting Between Locations
```
Function shortcut_to_location(target_memory):
  // Vector arithmetic: current_position + stored_vector = shortcut
  activate_memory(target_memory)
  
  While not at_target():
    // Steering automatically handles vector addition
    execute_steering_step()
    update_position()
    
  deactivate_memory(target_memory)
```

### Route Learning and Optimization
```
Function learn_route():
  discovered_locations = []
  
  While not all_locations_found():
    If no_memories_available():
      perform_random_walk()
    Else:
      current_memory = select_next_memory()
      navigate_using_memory(current_memory)
    
    If new_location_discovered():
      store_vector_memory()
      add to discovered_locations
    
    return_to_nest()
    update_memory_accuracy()  // Refine memories with each visit
```

## Utility Functions

### Sigmoid Activation
```
Function sigmoid(input, slope=1, offset=0):
  return 1 / (1 + exp(-(slope * input - offset)))
```

### Distance Calculation
```
Function calculate_vector_distance(CPU4_pattern):
  max_activity = max(CPU4_pattern)
  min_activity = min(CPU4_pattern)
  return max_activity - min_activity
```

### Search Behavior
```
Function systematic_search():
  // Emerges naturally when home vector approaches zero
  While vector_amplitude > threshold:
    continue_current_direction()
  
  // When amplitude near zero, noise dominates -> search pattern
  perform_random_search()
```

## Parameters

### Neural Parameters
- Sigmoid slope: a = 1.0
- Sigmoid offset: b = 0.0
- Noise standard deviation: σ = 0.1
- Accumulation rate: acc = 0.0025
- Decay rate: decay = 0.1
- Mutual inhibition strength: d = 0.33

### Behavioral Parameters
- Maximum steps per trip: 10,000
- Feeder catchment radius: 20 steps
- Speed: 0.15 units/step
- Turn noise concentration: κ = 100.0
- Recalibration efficiency: 0.0 to 1.0 (adjustable)

## Implementation Notes

1. **Modular Design**: Each neural layer can be implemented as separate modules
2. **Parallel Processing**: CPU4 neurons can be updated in parallel
3. **Memory Management**: Implement efficient storage/retrieval for multiple vector memories
4. **Noise Handling**: Add Gaussian noise to neural outputs for realistic behavior
5. **Boundary Conditions**: Clip neuron activities to [0,1] range
6. **Optimization**: Use lookup tables for trigonometric functions if needed

## Applications

This algorithm can be applied to:
- Autonomous robot navigation
- Swarm robotics path planning  
- Bio-inspired navigation systems
- Multi-agent coordination
- Foraging optimization problems

## Extensions

Potential enhancements include:
- Integration with landmark-based navigation
- Adaptive learning rates
- Dynamic environment handling
- Social information integration
- Hierarchical route planning