#pragma once
#include "rl_agent.h"
#include "dq_network.h"
#include <torch/torch.h>

class DqnAgent : public ReinforcementLearningAgent {
protected:
    torch::nn::Sequential model_network;
    torch::nn::Sequential target_network;
    int num_actions = -1;
    torch::optim::Optimizer* optimiser;
    int target_update_frequency;
    float target_update_tau;
    float initial_epsilon;
    float final_epsilon;
    int epsilon_steps;
    float epsilon;
    int epsilon_update_freq;
    int epsilon_log_freq;
public:
    DqnAgent(
        torch::nn::Sequential network,
        int target_update_frequency = 150,
        float target_update_tau = 1.0f,
        float initial_epsilon = 0.985f,
        float final_epsilon = 0.1f,
        int epsilon_steps = 400,
        int epsilon_update_freq = 50,
        int epsilon_log_freq = 800
    );

    void update_epsilon();
    void save(const string& location = "training\\results\\checkpoints");
};

class TaskPricingDqnAgent : public DqnAgent, public TaskPricingAgent {
protected:
    SampleBuf replay_buffer;
    static const int network_obs_width = 10;
    float failed_auction_reward;
    float failed_auction_reward_multiplier;
public:
    TaskPricingDqnAgent(
        int agent_name = -1,
        torch::nn::Sequential network = create_lstm_dq_network(network_obs_width, 31),
        int epsilon_steps = 140000,
        float failed_auction_reward = -0.05f,
        float failed_auction_reward_multiplier = -1.5f,
        float reward_scaling = 0.4f
    );

    torch::Tensor _train(torch::Tensor states, torch::Tensor actions, torch::Tensor next_states,
        torch::Tensor rewards, torch::Tensor dones);
    static StateList network_obs(const Task& auction_task, const TaskList& allocated_tasks,
        const Server& server, int time_step);
    static torch::Tensor network_obs_tensor(const Task& auction_task, const TaskList& allocated_tasks, 
        const Server& server, int time_step);
    void winning_auction_bid(const TaskPricingState& agent_state, const Action& action,
        const Task& finished_task, const TaskPricingState& next_agent_state);
    void failed_auction_bid( const TaskPricingState& agent_state, const Action& action, 
        const TaskPricingState& next_agent_state);
    float get_action(const Task& auction_task, const TaskList& allocated_tasks, const Server& server,
        int time_step, bool training = false);
    void train();
    void add_trajectory(const StateList& state, const Action& action, const StateList& next_state,
        const Reward& reward, const Done& done = false);
};

class ResourceWeightingDqnAgent : public DqnAgent, public ResourceWeightingAgent {
protected:
    SampleRWBuf replay_buffer;
    static const int network_obs_width = 10;
    float other_task_discount;
    float success_reward;
    float failed_reward;
public:
    ResourceWeightingDqnAgent(
        int agent_name = -1,
        torch::nn::Sequential network = create_bidirectional_dq_network(network_obs_width, 21),
        int epsilon_steps = 100000,
        float other_task_discount = 0.4f,
        float success_reward = 1.0f,
        float failed_reward = -1.5f
    );

    torch::Tensor _train(torch::Tensor states, torch::Tensor actions, torch::Tensor next_states,
        torch::Tensor rewards, torch::Tensor dones);
    static StateList network_obs(const TaskList& allocated_tasks, const Server& server, int time_step);
    void resource_allocation_obs(const ResourceAllocationState& agent_state, const TaskWeightMap& actions,
        const ResourceAllocationState& next_agent_state, const TaskList& finished_tasks);
    TaskWeightMap get_actions(const TaskList& allocated_tasks, const Server& server,
        int time_step, bool training = false);
    void train();
    void add_trajectory(const StateList& state, const ActionList& action, const StateList& next_state,
        const RewardList& reward, const DoneList& done);
};