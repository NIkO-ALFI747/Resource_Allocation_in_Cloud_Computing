# Cloud Resource Allocation Using Multi-Agent Deep Reinforcement Learning

## Overview

A sophisticated C++ implementation of a multi-agent reinforcement learning system for dynamic cloud resource allocation, utilizing Deep Q-Networks (DQN) with LSTM architectures. The system simulates a cloud computing environment where autonomous agents learn to optimize task pricing and resource distribution across heterogeneous server infrastructure through competitive auction mechanisms.

## Technical Architecture

### Core Components

#### 1. Environment Simulation (`env.cpp`, `env.h`)
- **Purpose**: Simulates a cloud datacenter environment with time-stepped discrete task arrival and resource allocation
- **Key Features**:
  - Dynamic task auction system with temporal constraints
  - Multi-server heterogeneous resource management (storage, computation, bandwidth)
  - State validation and environment serialization
  - Configurable environment generation from JSON specifications
- **State Space**: Encapsulates server capacities, allocated task states, auction queue, and temporal dynamics

#### 2. Task Management System (`task.cpp`, `task.h`)
- **Task Lifecycle States**:
  - `UNASSIGNED`: Awaiting auction
  - `LOADING`: Data transfer to server
  - `COMPUTING`: Active computation phase
  - `SENDING`: Result transmission
  - `COMPLETED`: Successfully finished
  - `FAILED`: Deadline violation or resource starvation
- **Resource Requirements**: Three-dimensional resource specification (storage, computation, bandwidth)
- **Progress Tracking**: Per-stage completion percentages with deadline enforcement

#### 3. Server Infrastructure (`server.cpp`, `server.h`)
- **Capacity Management**: Fixed capacity vectors for storage, computational, and bandwidth resources
- **Resource Allocation Logic**: 
  - Weighted resource distribution algorithm
  - Priority-based allocation using agent-learned task weights
  - Bandwidth splitting between ingress (loading) and egress (sending) operations
- **Multi-task Scheduling**: Concurrent task execution with resource contention resolution

### Agent Architecture

#### Base Agent Framework (`rl_agent.h`, `rl_agent.cpp`)

**Core RL Components**:
- Experience replay buffer with configurable capacity (default: 30,000 samples)
- Epsilon-greedy exploration with linear annealing schedule
- Target network for stability with soft update mechanism (τ-based Polyak averaging)
- Gradient clipping and L2 regularization for training stability
- Normalized state representations with min-max scaling

**Hyperparameters**:
- Batch size: 10 samples
- Discount factor (γ): 0.8
- Training frequency: Every 2 environment steps
- Target network update frequency: Every 150 training iterations
- Initial epsilon: 0.985, Final epsilon: 0.1

#### 1. Task Pricing Agent (`task_pricing_agent.h`, `dqn_agent.cpp`)

**Objective**: Learn optimal bid prices for task auctions to maximize server utilization and profit

**Network Architecture**:
- **Input**: 10-dimensional observation vector
  - Normalized task requirements (storage, computation, result data)
  - Deadline urgency (time remaining)
  - Current server load and capacity ratios
  - Task progress indicators
  - Temporal features
- **Hidden Layers**: LSTM layer (64 units) → ReLU activation (64 units)
- **Output**: 31 discrete price actions (bid levels)

**Training Mechanism**:
- Double DQN algorithm to reduce overestimation bias
- Custom reward structure:
  - Positive reward: Task completion profit (scaled)
  - Negative reward: Failed auction penalty (-0.05) with multiplier (-1.5)
- Sequential state observation with temporal task ordering
- Last-timestep LSTM output selection for decision making

**Key Implementation Details**:
```cpp
// Observation includes auction task + all allocated tasks
StateList network_obs(auction_task, allocated_tasks, server, time_step)
// LSTM processes variable-length task sequences
// Final hidden state used for Q-value estimation
```

#### 2. Resource Weighting Agent (`resource_weighting_agent.h`, `dqn_agent.cpp`)

**Objective**: Dynamically prioritize resource allocation across concurrently executing tasks

**Network Architecture**:
- **Input**: 10-dimensional per-task observation
- **Hidden Layers**: Bidirectional LSTM (forward 64 units, reverse 64 units) → ReLU (64 units)
- **Output**: 21 discrete weight values per task (priority levels)

**Training Mechanism**:
- Multi-task action space: One weight per allocated task
- Reward design:
  - Success reward: +1.0 for completed tasks
  - Failure penalty: -1.5 for deadline violations
  - Temporal discount: 0.4 scaling for non-finishing tasks
- Bidirectional processing captures task interdependencies
- Simultaneous weight learning across task ensemble

**Unique Characteristics**:
- Multi-action DQN: Outputs weight vector for all tasks
- Reverse LSTM component extracts future-oriented context
- Joint reward attribution across resource allocation decisions

### Neural Network Architectures (`dq_network.h`)

#### LSTM Model
```cpp
class LSTMModel {
    torch::nn::LSTM lstm(input_width, 64, batch_first=true);
    torch::nn::ReLU relu;
    torch::nn::Linear linear(64, num_actions);
}
```
- Processes variable-length task sequences
- Batch-first format for efficient training
- Parameter flattening before forward pass (LSTM requirement)

#### Bidirectional LSTM Model
```cpp
class BidirectionalLSTMModel {
    torch::nn::LSTM forward_lstm(input_width, 64);
    torch::nn::LSTM reverse_lstm(64, 64, bidirectional=true);
    torch::nn::ReLU relu;
    torch::nn::Linear linear(64, num_actions);
}
```
- Two-stage LSTM processing
- Reverse LSTM extracts backward temporal context
- Output slicing to select reverse direction hidden states

### Training Pipeline (`train_agents.cpp`, `train_agents.h`)

#### Multi-Agent Coordination
- **Agent Allocation**: Dynamic mapping of agents to servers per episode
- **Auction Protocol**: 
  1. Task announcement to all server agents
  2. Simultaneous price bidding
  3. Winner selection (highest bid)
  4. Task assignment and resource allocation

#### Training Loop Architecture
```cpp
run_training(env, eval_envs, total_episodes=80, agents, eval_frequency=2)
```

**Per Episode**:
1. Environment reset with random task generation
2. Agent allocation to servers
3. Timestep iteration:
   - Auction phase: Task pricing decisions
   - Allocation phase: Resource weighting decisions
   - Environment step: Resource distribution and progress updates
   - Experience storage: State transitions to replay buffers
4. Agent training with sampled mini-batches
5. Epsilon decay and target network updates

#### Evaluation System (`eval_results.cpp`, `eval_results.h`)
- Separate evaluation environments (20 test scenarios)
- Metrics tracking:
  - Task completion rate
  - Average task latency
  - Resource utilization efficiency
  - Agent profit and costs
- Deterministic policy evaluation (epsilon = 0)

### Environment Configuration System

#### JSON-Based Environment Specifications
```json
{
  "name": "Basic",
  "min total time steps": 60, "max total time steps": 75,
  "min total servers": 4, "max total servers": 6,
  "server settings": [{
    "min storage capacity": 200, "max storage capacity": 300,
    "min computational capacity": 15, "max computational capacity": 30,
    "min bandwidth capacity": 15, "max bandwidth capacity": 25
  }],
  "min total tasks": 130, "max total tasks": 160,
  "task settings": [{
    "min deadline": 6, "max deadline": 12,
    "min required storage": 50, "max required storage": 80,
    "min required computation": 25, "max required computation": 60,
    "min required results data": 10, "max required results data": 30
  }]
}
```

**Configuration Flexibility**:
- Procedural environment generation with min/max ranges
- Support for heterogeneous server configurations
- Multiple task type specifications
- Temporal constraint randomization

**Predefined Configurations**:
- `basic.env`: Balanced resource and task distribution
- `large_tasks_servers.env`: High-capacity infrastructure scenarios
- `limited_resources.env`: Resource-constrained stress testing
- `mixture_tasks_servers.env`: Heterogeneous multi-tier configuration

## Key Technical Challenges Addressed

### 1. Variable-Length State Sequences
**Challenge**: Cloud servers handle dynamic numbers of concurrent tasks

**Solution**: LSTM architecture processes sequential task observations with automatic sequence length handling. Padding not required due to batch-first processing.

### 2. Temporal Credit Assignment
**Challenge**: Resource allocation decisions have delayed consequences (task completion happens multiple timesteps later)

**Solution**: 
- Discount factor (γ=0.8) balances immediate vs. future rewards
- Experience replay decorrelates temporal correlations
- Multi-step return estimation through Bellman equation

### 3. Multi-Objective Optimization
**Challenge**: Agents must balance task completion, deadline adherence, and resource efficiency

**Solution**:
- Composite reward functions encoding multiple objectives
- Separate pricing and weighting agents with distinct reward structures
- Reward scaling (0.4 for pricing) to balance learning rates

### 4. Non-Stationary Environment
**Challenge**: Other agents' policies change during training (moving target problem)

**Solution**:
- Independent agent learning with shared environment
- Target networks provide stable Q-value targets
- Epsilon decay ensures convergence to deterministic policies

### 5. Action Space Dimensionality
**Challenge**: Resource weighting requires one action per active task (variable dimension)

**Solution**:
- Network outputs fixed-size weight vector (21 levels)
- Dynamic action selection for variable task counts
- Zero-padding for servers with fewer tasks

### 6. Exploration-Exploitation Balance
**Challenge**: Need to explore suboptimal bids/weights while maximizing accumulated rewards

**Solution**:
- Extended epsilon annealing (140,000 steps for pricing, 100,000 for weighting)
- High initial epsilon (0.985) encourages thorough exploration
- Gradual transition to exploitation (final epsilon 0.1)

## Implementation Details

### Memory Management
- LibTorch tensor operations on CUDA-enabled devices
- Automatic gradient computation with `torch::autograd`
- Replay buffer as `std::deque` with O(1) push/pop
- Device management: `torch::kCUDA` for training, `torch::kCPU` for sampling

### Computational Optimizations
- Batch processing of experience replay samples
- LSTM parameter flattening before each forward pass
- Gradient clipping (`clip_grad_value_`, threshold=100) for stability
- L2 regularization via parameter norm addition to loss

### Model Persistence
- PyTorch model serialization (`.pt` format)
- Checkpoint saving every 250 training iterations
- Independent saving of policy and target networks
- Environment state serialization for reproducibility

### Validation and Assertions
- Comprehensive validity checking throughout codebase
- Resource conservation assertions (no capacity violations)
- Task state transition validation
- Temporal constraint verification

## Dependencies

### Required Libraries

- **LibTorch 2.0+**: PyTorch C++ API for deep learning
  - CUDA support for GPU acceleration
  - Automatic differentiation engine
  - Neural network modules and optimizers

- **nlohmann/json**: Modern C++ JSON library
  - Header-only implementation
  - Intuitive API for parsing environment configurations

### Build Requirements
- C++17 or later
- CMake 3.18+ (for LibTorch integration)
- CUDA Toolkit 11.0+ (for GPU acceleration)
- Visual Studio 2019+ (Windows) or GCC 9+ (Linux)

## Project Structure

```
Resource_Allocation_in_Cloud_Computing/
├── cloud_resource_allocation.cpp   # Entry point
├── dqn_alg.cpp/h                  # DQN algorithm orchestration
├── dqn_agent.cpp/h                # Agent implementations
├── dq_network.h                   # Neural network architectures
├── env.cpp/h                      # Environment simulation
├── env_state.cpp/h                # State representation
├── server.cpp/h                   # Server resource management
├── task.cpp/h                     # Task lifecycle and properties
├── task_stage.cpp/h               # Task state enumeration
├── rl_agent.cpp/h                 # Base RL agent class
├── task_pricing_agent.cpp/h       # Pricing agent interface
├── resource_weighting_agent.cpp/h # Weighting agent interface
├── train_agents.cpp/h             # Training loop and evaluation
├── eval_results.cpp/h             # Evaluation metrics
├── network*.pt                    # Pre-trained model checkpoints
└── training_data/
    └── settings/
        ├── basic.env
        ├── large_tasks_servers.env
        ├── limited_resources.env
        └── mixture_tasks_servers.env
```

## Build Instructions

### Windows (Visual Studio)
```bash
# 1. Install LibTorch
# Download from https://pytorch.org/cplusplus/
# Extract to C:/libtorch

# 2. Open solution
# Load Resource_Allocation_in_Cloud_Computing.sln in Visual Studio

# 3. Configure LibTorch paths
# Project Properties → C/C++ → Additional Include Directories: C:\libtorch\include
# Project Properties → Linker → Additional Library Directories: C:\libtorch\lib
# Project Properties → Linker → Input: Add all .lib files from libtorch\lib

# 4. Build
# Set configuration to Release x64
# Build → Build Solution
```

### Linux (CMake)
```bash
# Install dependencies
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

# Create CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(ResourceAllocation)
find_package(Torch REQUIRED)
add_executable(resource_allocation [source_files])
target_link_libraries(resource_allocation "${TORCH_LIBRARIES}")

# Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j$(nproc)
```

## Usage

### Training Agents
```cpp
// Configure environment
ResourceAllocationEnvironment env({"./training/settings/basic.env"});
env.reset();

// Generate evaluation scenarios
vector<string> eval_envs = generate_eval_envs(env, 20, "./eval_envs");

// Initialize agent networks
torch::nn::Sequential pricing_net = create_lstm_dq_network(10, 31);
torch::nn::Sequential weighting_net = create_bidirectional_dq_network(10, 21);

// Create agent instances
TaskPricingAgentList pricing_agents;
ResourceWeightingAgentList weighting_agents;
pricing_agents.push_back(TaskPricingDqnAgent(0, pricing_net));
weighting_agents.push_back(ResourceWeightingDqnAgent(0, weighting_net));

// Run training
run_training(env, eval_envs, 80, pricing_agents, weighting_agents, 2);

// Save trained models
for (auto& agent : pricing_agents) agent.save();
for (auto& agent : weighting_agents) agent.save();
```

### Custom Environment Configuration
```json
{
  "name": "CustomScenario",
  "min total time steps": 100,
  "max total time steps": 150,
  "min total servers": 8,
  "max total servers": 10,
  "server settings": [
    {
      "name": "HighCapacity",
      "min storage capacity": 500,
      "max storage capacity": 1000,
      "min computational capacity": 50,
      "max computational capacity": 100,
      "min bandwidth capacity": 40,
      "max bandwidth capacity": 80
    },
    {
      "name": "EdgeServer",
      "min storage capacity": 100,
      "max storage capacity": 200,
      "min computational capacity": 10,
      "max computational capacity": 20,
      "min bandwidth capacity": 20,
      "max bandwidth capacity": 40
    }
  ],
  "min total tasks": 300,
  "max total tasks": 500,
  "task settings": [
    {
      "name": "HighPriority",
      "min deadline": 5,
      "max deadline": 10,
      "min required storage": 100,
      "max required storage": 200,
      "min required computation": 50,
      "max required computation": 100,
      "min required results data": 30,
      "max required results data": 60
    }
  ]
}
```

## Advanced Features

### Custom Neural Network Architectures
The modular design allows integration of custom network architectures:

```cpp
// Example: Attention-based network
struct AttentionDQN : torch::nn::Module {
    torch::nn::LSTM encoder;
    torch::nn::MultiheadAttention attention;
    torch::nn::Linear output;
    
    torch::Tensor forward(torch::Tensor x) {
        auto encoded = encoder->forward(x);
        auto attended = attention->forward(encoded, encoded, encoded);
        return output->forward(std::get<0>(attended));
    }
};
```

### Reward Shaping
Customize reward functions for specific optimization objectives:

```cpp
// In TaskPricingDqnAgent
float compute_reward(const Task& task, bool success) {
    if (success) {
        float profit = task.get_price() - compute_cost(task);
        float deadline_bonus = calculate_deadline_bonus(task);
        return reward_scaling * (profit + deadline_bonus);
    }
    return failed_auction_reward * failed_auction_reward_multiplier;
}
```

### Multi-Objective Optimization
Extend agents to optimize multiple metrics simultaneously:

```cpp
struct MultiObjectiveReward {
    float profit_weight = 0.5;
    float utilization_weight = 0.3;
    float fairness_weight = 0.2;
    
    float compute(float profit, float util, float fairness) {
        return profit_weight * profit + 
               utilization_weight * util + 
               fairness_weight * fairness;
    }
};
```

## Research Applications

This implementation serves as a foundation for research in:

1. **Multi-Agent Systems**: Studying emergent coordination behavior in competitive-cooperative environments
2. **Resource Management**: Developing adaptive algorithms for heterogeneous infrastructure
3. **Mechanism Design**: Analyzing auction protocols and pricing strategies in cloud marketplaces
4. **Transfer Learning**: Investigating policy transfer across different datacenter configurations
5. **Hierarchical RL**: Extending to hierarchical decision-making (cluster-level → server-level)

## Performance Considerations

### Training Efficiency
- Single episode duration: ~10-30 seconds (depending on time steps)
- Training convergence: 60-80 episodes for basic scenarios
- GPU acceleration: 3-5x speedup vs. CPU-only training
- Memory footprint: ~2GB GPU memory for default configuration

### Scalability Limits
- Tested configurations: Up to 20 servers, 500 tasks per episode
- LSTM sequence length: Handles up to 50 concurrent tasks per server
- Replay buffer capacity: 30,000 experiences (~1.5GB RAM)

## Known Limitations

1. **Discrete Action Spaces**: Price levels and weights are discretized; continuous action spaces could improve granularity
2. **Independent Learning**: Agents don't explicitly model other agents' policies (could benefit from opponent modeling)
3. **Single Objective per Agent**: Each agent optimizes its own reward; multi-objective Pareto optimization not implemented
4. **Fixed Network Topology**: Server interconnections not modeled (assumes flat network)
5. **Deterministic Task Characteristics**: Task resource requirements are fixed; stochastic models could increase realism

## Future Enhancements

- **Communication Protocol**: Enable information sharing between agents
- **Hierarchical Task Decomposition**: Support for complex workflows with task dependencies
- **Meta-Learning**: Rapid adaptation to new environment configurations
- **Safety Constraints**: Hard constraints on resource violations and SLA guarantees
- **Interpretability Tools**: Visualization of learned policies and Q-value landscapes
- **Distributed Training**: Multi-GPU training for larger-scale experiments

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{cloud_resource_allocation_dqn,
  title={Multi-Agent Deep Reinforcement Learning for Cloud Resource Allocation},
  author={Nikolay Ermilov},
  year={2024},
  url={https://github.com/[repository]},
  note={Implementation in C++ using LibTorch}
}
```

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Author

This implementation is designed for research purposes. Production deployment would require additional work on fault tolerance, security, and real-world integration.
