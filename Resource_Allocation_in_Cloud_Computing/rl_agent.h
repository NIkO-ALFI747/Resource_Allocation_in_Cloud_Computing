#pragma once
#include "task_pricing_agent.h"
#include "resource_weighting_agent.h"
#include <functional>
#include <torch/torch.h>
struct Sample;
struct SampleRW;

typedef vector<float> State;
typedef float Action, Reward;
typedef bool Done;
typedef vector<Action> ActionList;
typedef vector<State> StateList;
typedef vector<Reward> RewardList;
typedef vector<Done> DoneList;
typedef function<torch::Tensor(torch::Tensor, torch::Tensor)> LossFunc;
typedef deque<Sample> SampleBuf;
typedef deque<SampleRW> SampleRWBuf;
inline torch::Device device_(torch::kCUDA);
inline torch::Device cpu_(torch::kCPU);

class TaskPricingState {
public:
    Task auction_task;
    TaskList allocated_tasks;
    Server server;
    int time_step;
    TaskPricingState(const Task& auction_task, const TaskList& allocated_tasks, const Server& server, int time_step) :
        auction_task(auction_task),
        allocated_tasks(allocated_tasks),
        server(server),
        time_step(time_step)
    {}
};
class ResourceAllocationState {
public:
    TaskList allocated_tasks;
    Server server;
    int time_step;
    ResourceAllocationState(const TaskList& allocated_tasks, const Server& server, int time_step) :
        allocated_tasks(allocated_tasks),
        server(server),
        time_step(time_step)
    {}
};
struct Sample {
    StateList state;
    Action action;
    StateList next_state;
    Reward reward;
    Done done;
};
struct SampleRW {
    StateList state;
    ActionList action;
    StateList next_state;
    RewardList reward;
    DoneList done;
};

class ReinforcementLearningAgent {
protected:
    int batch_size;
    LossFunc error_loss_fn;
    int initial_training_replay_size;
    int training_freq;
    float discount_factor;
    int replay_buffer_length;
    int save_frequency;
    string save_folder;
    int training_loss_log_freq;
    float reward_scaling;
    int total_actions;
    int total_updates;
    int total_observations;

    virtual torch::Tensor _train(torch::Tensor states, torch::Tensor actions, torch::Tensor next_states,
        torch::Tensor rewards, torch::Tensor dones) = 0;
public:
    virtual void save(const string& location = "training\\results\\checkpoints") = 0;
    static State normalise_task(const Task& task, const Server& server, int time_step);
    void update_target_network(torch::nn::Sequential& model_network, torch::nn::Sequential& target_network, float tau);

    ReinforcementLearningAgent(
        int batch_size = 10,
        LossFunc error_loss_fn = nullptr,
        int initial_training_replay_size = 800,
        int training_freq = 2,
        float discount_factor = 0.8f,
        int replay_buffer_length = 30000,
        int save_frequency = 250,
        string save_folder = "checkpoint",
        int training_loss_log_freq = 250,
        float reward_scaling = 1.0f
    );
};
