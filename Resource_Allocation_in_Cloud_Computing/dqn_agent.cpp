#pragma once
#include <cassert>
#include <filesystem>
#include <random>
#include <torch/torch.h>
#include "dq_network.h"
#include "dqn_agent.h"

/// DqnAgent

DqnAgent::DqnAgent(
    torch::nn::Sequential network,
    int target_update_frequency,
    float target_update_tau,
    float initial_epsilon,
    float final_epsilon,
    int epsilon_steps,
    int epsilon_update_freq,
    int epsilon_log_freq
) 
    : ReinforcementLearningAgent() 
{
    if (network->is_empty()) {
        cout << "ѕараметры нейронной сети не заданы!\n";
    }
    if (network->children().back()->parameters().empty()) {
        cout << "ѕараметры выходного сло€ не заданы!\n";
    }
    model_network = network;
    target_network = network;
    num_actions = network->children().back()->parameters().back().sizes()[0];
    model_network->to(device_);
    target_network->to(device_);
    optimiser = new torch::optim::Adam(model_network->parameters());
    this->target_update_frequency = target_update_frequency;
    this->target_update_tau = target_update_tau;
    this->initial_epsilon = initial_epsilon;
    this->final_epsilon = final_epsilon;
    this->epsilon_steps = epsilon_steps;
    epsilon = initial_epsilon;
    this->epsilon_update_freq = epsilon_update_freq;
    this->epsilon_log_freq = epsilon_log_freq;
}

void DqnAgent::update_epsilon() {
    total_actions++;
    if (total_actions % epsilon_update_freq == 0) {
        epsilon = max(
            (final_epsilon - initial_epsilon) * total_actions / epsilon_steps + initial_epsilon,
            final_epsilon
        );
        if (total_actions % epsilon_log_freq == 0) {
            cout << "Epsilon агента: " << epsilon << "\n"
                 << "ќбщее кол-во действий: " << total_actions << "\n";
        }
    }
}
void DqnAgent::save(const string& location) {
    string path = filesystem::current_path().string() + "\\" + location + "\\" + save_folder;
    if (!filesystem::exists(path))
        filesystem::create_directories(path);
    string file_name = path + "\\update_" + to_string(total_updates) + ".pt";
    torch::save(model_network, file_name);
}
torch::Tensor TaskPricingDqnAgent::_train(
    torch::Tensor states,
    torch::Tensor actions,
    torch::Tensor next_states,
    torch::Tensor rewards,
    torch::Tensor dones
) {
    actions = actions.to(torch::kLong);
    torch::Tensor state_q_values = model_network->forward(states).to(cpu_);
    int last_index = state_q_values.size(1) - 1;
    torch::Tensor states_actions_q_values = state_q_values.narrow(1, last_index, 1).squeeze(1);
    torch::Tensor states_actions_q_values2 = torch::gather(states_actions_q_values, 1, actions.unsqueeze(1)).squeeze(0).to(device_);
    torch::autograd::GradMode::set_enabled(false);
    torch::Tensor next_state_q_values = target_network->forward(next_states).to(cpu_);
    torch::autograd::GradMode::set_enabled(true);
    torch::Tensor next_actions = torch::argmax(next_state_q_values, 2);
    last_index = next_actions.size(1) - 1;
    next_actions = next_actions.narrow(1, last_index, 1).squeeze(1);
    torch::Tensor next_states_actions_q_values = next_state_q_values.narrow(1, last_index, 1).squeeze(1);
    torch::Tensor next_states_actions_q_values2 = torch::gather(next_states_actions_q_values, 1, next_actions.unsqueeze(1)).squeeze().to(device_);
    torch::Tensor disc_tensor = torch::tensor(discount_factor, device_);
    torch::Tensor dones_tensor = torch::tensor(1.0f).to(device_) - dones;
    torch::Tensor target = rewards + (disc_tensor * next_states_actions_q_values2 * dones_tensor);
    target.detach_();
    torch::Tensor loss = error_loss_fn(target, states_actions_q_values2);
    for (const auto& param : model_network->parameters())
        loss += param.norm(2);
    optimiser->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_value_(model_network->parameters(), 100);
    optimiser->step();
    if (total_updates % target_update_frequency == 0)
        update_target_network(model_network, target_network, target_update_tau);
    total_updates++;
    return loss;
}

/// TaskPricingDqnAgent

TaskPricingDqnAgent::TaskPricingDqnAgent(
    int agent_name,
    torch::nn::Sequential network,
    int epsilon_steps,
    float failed_auction_reward,
    float failed_auction_reward_multiplier,
    float reward_scaling
)
    : TaskPricingAgent("Task pricing DQN agent" + to_string(agent_name)), DqnAgent(network)
{
    this->reward_scaling = reward_scaling;
    if (failed_auction_reward > 0.0f || failed_auction_reward_multiplier > 0.0f) {
        cout << "¬ознаграждение или множитель за неудачный аукцион должен быть не положительным!\n";
        assert(false);
    }
    this->failed_auction_reward = failed_auction_reward;
    this->failed_auction_reward_multiplier = failed_auction_reward_multiplier;
    save_folder += "_task_pricing";
    int input_layer_width = network->children().front()->parameters().front().size(1);
    if (input_layer_width != network_obs_width) {
        cout << "–азмер входа сети не соответствует заданному!\n";
        assert(false);
    }
    this->epsilon_steps = epsilon_steps;
}

StateList TaskPricingDqnAgent::network_obs(
    const Task& auction_task,
    const TaskList& allocated_tasks,
    const Server& server,
    int time_step
) {
    StateList observation;
    State auction_task_norm = normalise_task(auction_task, server, time_step);
    auction_task_norm.push_back(1.0f);
    for (const auto& task : allocated_tasks) {
        State task_norm = normalise_task(task, server, time_step);
        task_norm.push_back(0.0f);
        observation.push_back(task_norm);
    }
    observation.push_back(auction_task_norm);
    return observation;
}
torch::Tensor TaskPricingDqnAgent::network_obs_tensor(
    const Task& auction_task,
    const TaskList& allocated_tasks,
    const Server& server,
    int time_step
) {
    vector<torch::Tensor> tensor_sequence;
    State auction_task_norm = normalise_task(auction_task, server, time_step);
    auction_task_norm.push_back(1.0f);
    torch::Tensor state_tensor = torch::from_blob(auction_task_norm.data(), { static_cast<long>(auction_task_norm.size()) }, torch::kFloat32).clone();
    for (const auto& task : allocated_tasks) {
        State task_norm = normalise_task(task, server, time_step);
        task_norm.push_back(0.0f);
        torch::Tensor state_tensor = torch::from_blob(task_norm.data(), { static_cast<long>(task_norm.size()) }, torch::kFloat32).clone();
        tensor_sequence.push_back(state_tensor);
    }
    tensor_sequence.push_back(state_tensor);
    torch::Tensor obs_tensor = torch::stack(tensor_sequence, 0);
    obs_tensor = obs_tensor.unsqueeze(0);
    return obs_tensor;
}
void TaskPricingDqnAgent::winning_auction_bid(
    const TaskPricingState& agent_state,
    const Action& action,
    const Task& finished_task,
    const TaskPricingState& next_agent_state
) {
    if (action < 0) {
        cout << "ƒействие (цена) задачи не должно быть отрицательным!\n";
        assert(false);
    }
    if (finished_task.get_stage() != TaskStage::COMPLETED && finished_task.get_stage() != TaskStage::FAILED) {
        cout << "«авершенные задачи должны иметь завершенное состо€ние!\n";
        assert(false);
    }
    Reward reward = finished_task.get_price() * (finished_task.get_stage() == TaskStage::COMPLETED ? 1.0f : failed_auction_reward_multiplier);
    StateList obs = network_obs(agent_state.auction_task, agent_state.allocated_tasks, agent_state.server, agent_state.time_step);
    StateList next_obs = network_obs(next_agent_state.auction_task, next_agent_state.allocated_tasks, next_agent_state.server, next_agent_state.time_step);
    add_trajectory(obs, action, next_obs, reward);
}
void TaskPricingDqnAgent::failed_auction_bid(
    const TaskPricingState& agent_state,
    const Action& action,
    const TaskPricingState& next_agent_state
) {
    if (action < 0) {
        cout << "ƒействие (цена) задачи не должно быть отрицательным!\n";
        assert(false);
    }
    StateList obs = network_obs(agent_state.auction_task, agent_state.allocated_tasks, agent_state.server, agent_state.time_step);
    StateList next_obs = network_obs(next_agent_state.auction_task, next_agent_state.allocated_tasks, next_agent_state.server, next_agent_state.time_step);
    add_trajectory(obs, action, next_obs, action == 0.0f ? 0.0f : failed_auction_reward);
}
float TaskPricingDqnAgent::get_action(
    const Task& auction_task,
    const TaskList& allocated_tasks,
    const Server& server,
    int time_step,
    bool training
) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    if (training) {
        update_epsilon();
        if (dis(gen) < epsilon) {
            auto auction_act = rand() % num_actions;
            return auction_act;
        }
    }
    torch::Tensor obs_tensor = network_obs_tensor(auction_task, allocated_tasks, server, time_step).to(device_);
    torch::Tensor q_values = model_network->forward(obs_tensor).to(cpu_);
    torch::Tensor action = torch::argmax(q_values, 2, false);
    int last_index = action.size(1) - 1;
    float last_element = action[0][last_index].item<float>();
    return last_element;
}
void TaskPricingDqnAgent::train() {
    SampleBuf samples;
    random_device rd;
    mt19937 gen(rd());
    vector<int> indices(replay_buffer.size());
    for (int i = 0; i < indices.size(); ++i)
        indices[i] = i;
    shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < batch_size; ++i)
        samples.push_back(replay_buffer.at(indices[i]));
    torch::Tensor training_loss;
    for (auto sequence : samples) {
        vector<float> actions, rewards, dones;
        vector<torch::Tensor> states, next_states;
        vector<torch::Tensor> tensor_sequence_list;
        for (auto state_el : sequence.state) {
            vector<float> state = state_el;
            torch::Tensor state_tensor = torch::from_blob(state.data(), { static_cast<long>(state.size()) }, torch::kFloat32).clone();
            tensor_sequence_list.push_back(state_tensor);
        }
        torch::Tensor tensor_sequence = torch::stack(tensor_sequence_list, 0);
        tensor_sequence = tensor_sequence.unsqueeze(0).to(device_);
        vector<torch::Tensor> next_tensor_sequence_list;
        for (auto next_state_el : sequence.next_state) {
            vector<float> next_state = next_state_el;
            torch::Tensor next_state_tensor = torch::from_blob(next_state.data(), { static_cast<long>(next_state.size()) }, torch::kFloat32).clone();
            next_tensor_sequence_list.push_back(next_state_tensor);
        }
        torch::Tensor next_tensor_sequence = torch::stack(next_tensor_sequence_list, 0);
        next_tensor_sequence = next_tensor_sequence.unsqueeze(0).to(device_);
        actions.push_back(sequence.action);
        torch::Tensor actions_tensor = torch::from_blob(actions.data(), { static_cast<long>(actions.size()) }, torch::kFloat32).clone();
        rewards.push_back(sequence.reward);
        torch::Tensor rewards_tensor = torch::from_blob(rewards.data(), { static_cast<long>(rewards.size()) }, torch::kFloat32).clone().to(device_);
        dones.push_back(sequence.done ? 1.0f : 0.0f);
        torch::Tensor dones_tensor = torch::from_blob(dones.data(), { static_cast<long>(dones.size()) }, torch::kFloat32).clone().to(device_);
        training_loss = _train(tensor_sequence, actions_tensor, next_tensor_sequence, rewards_tensor, dones_tensor);
    }
    if (total_updates % training_loss_log_freq == 0) {
        cout << "ѕотери агента: \n" << training_loss << "\n"
            << "ќбщее кол-во наблюдений: " << total_observations << "\n";
    }
    if (total_updates % save_frequency == 0) {
        save();
    }
    total_updates++;
}
void TaskPricingDqnAgent::add_trajectory(
    const StateList& state,
    const Action& action,
    const StateList& next_state,
    const Reward& reward,
    const Done& done
) {
    Sample sample = { state, action, next_state, reward * reward_scaling, done };
    replay_buffer.push_back(sample);
    if (replay_buffer.size() > replay_buffer_length) replay_buffer.pop_front();
    total_observations++;
    if (total_observations >= initial_training_replay_size && total_observations % training_freq == 0)
        train();
}

/// ResourceWeightingDqnAgent

ResourceWeightingDqnAgent::ResourceWeightingDqnAgent(
    int agent_name,
    torch::nn::Sequential network,
    int epsilon_steps,
    float other_task_discount,
    float success_reward,
    float failed_reward
) :
    ResourceWeightingAgent("Resource weighting DQN agent" + to_string(agent_name)),
    DqnAgent(network)
{
    if (other_task_discount <= 0.0f) {
        cout << "—кидка при выполнении других задач должна быть положительна!\n";
        assert(false);
    }
    this->other_task_discount = other_task_discount;
    if (failed_reward >= 0.0f || success_reward <= 0.0f) {
        cout << "¬ознаграждение за проваленную задачу должно быть отрицательным, а за успешно завершенную - положительным!\n";
        assert(false);
    }
    this->success_reward = success_reward;
    this->failed_reward = failed_reward;
    save_folder += "_resource_weighting";
    this->epsilon_steps = epsilon_steps;
    int input_layer_width = network->children().front()->parameters().front().size(1);
    if (input_layer_width != network_obs_width) {
        cout << "–азмер входа сети не соответствует заданному!\n";
        assert(false);
    }
}

StateList ResourceWeightingDqnAgent::network_obs(
    const TaskList& allocated_tasks,
    const Server& server,
    int time_step
) {
    if (allocated_tasks.size() <= 1) {
        cout << "кол-во выполн€ющихс€ на сервере задач должно быть больше одной!\n";
        assert(false);
    }
    StateList obs;
    for (const auto& task : allocated_tasks) {
        State allocated_task_obs = normalise_task(task, server, time_step);
        allocated_task_obs.push_back(1.0f);
        obs.push_back(allocated_task_obs);
    }
    return obs;
}
torch::Tensor ResourceWeightingDqnAgent::_train(
    torch::Tensor states,
    torch::Tensor actions,
    torch::Tensor next_states,
    torch::Tensor rewards,
    torch::Tensor dones
) {
    actions = actions.to(torch::kLong);
    torch::Tensor state_q_values = model_network->forward(states).to(cpu_);
    torch::Tensor states_actions_q_values2 = torch::gather(state_q_values, 2, actions.unsqueeze(0).unsqueeze(2)).squeeze(2).squeeze(0).to(device_);
    torch::autograd::GradMode::set_enabled(false);
    torch::Tensor next_state_q_values = target_network->forward(next_states).to(cpu_);
    torch::autograd::GradMode::set_enabled(true);
    torch::Tensor next_actions = torch::argmax(next_state_q_values, 2);
    torch::Tensor next_states_actions_q_values = torch::gather(next_state_q_values, 2, next_actions.unsqueeze(2)).squeeze(2).squeeze(0).to(device_);
    torch::Tensor disc_tensor = torch::tensor(discount_factor, device_);
    torch::Tensor dones_tensor = torch::tensor(1.0f).to(device_) - dones;
    torch::Tensor target = rewards + (disc_tensor * next_states_actions_q_values * dones_tensor);
    target.detach_();
    torch::Tensor loss = error_loss_fn(target, states_actions_q_values2);
    for (const auto& param : model_network->parameters()) {
        loss += param.norm(2);
    }
    //cout << "loss: " << loss;
    optimiser->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_value_(model_network->parameters(), 100);
    optimiser->step();
    if (total_updates % target_update_frequency == 0)
        update_target_network(model_network, target_network, target_update_tau);
    total_updates++;
    return loss;
}
void ResourceWeightingDqnAgent::train() {
    SampleRWBuf samples;
    random_device rd;
    mt19937 gen(rd());
    vector<int> indices(replay_buffer.size());
    for (int i = 0; i < indices.size(); ++i)
        indices[i] = i;
    shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < batch_size; ++i)
        samples.push_back(replay_buffer.at(indices[i]));
    torch::Tensor training_loss;
    for (auto sequence : samples) {
        vector<float> actions, rewards, dones;
        vector<torch::Tensor> states, next_states;
        vector<torch::Tensor> tensor_sequence_list;
        for (auto state_el : sequence.state) {
            vector<float> state = state_el;
            torch::Tensor state_tensor = torch::from_blob(state.data(), { static_cast<long>(state.size()) }, torch::kFloat32).clone();
            tensor_sequence_list.push_back(state_tensor);
        }
        torch::Tensor tensor_sequence = torch::stack(tensor_sequence_list, 0);
        tensor_sequence = tensor_sequence.unsqueeze(0).to(device_);
        vector<torch::Tensor> next_tensor_sequence_list;
        for (auto next_state_el : sequence.next_state) {
            vector<float> next_state = next_state_el;
            torch::Tensor next_state_tensor = torch::from_blob(next_state.data(), { static_cast<long>(next_state.size()) }, torch::kFloat32).clone();
            next_tensor_sequence_list.push_back(next_state_tensor);
        }
        torch::Tensor next_tensor_sequence = torch::stack(next_tensor_sequence_list, 0);
        next_tensor_sequence = next_tensor_sequence.unsqueeze(0).to(device_);
        actions = sequence.action;
        torch::Tensor actions_tensor = torch::from_blob(actions.data(), { static_cast<long>(actions.size()) }, torch::kFloat32).clone();
        rewards = sequence.reward;
        torch::Tensor rewards_tensor = torch::from_blob(rewards.data(), { static_cast<long>(rewards.size()) }, torch::kFloat32).clone().to(device_);
        for (auto& done : sequence.done) {
            dones.push_back(done ? 1.0f : 0.0f);
        }
        torch::Tensor dones_tensor = torch::from_blob(dones.data(), { static_cast<long>(dones.size()) }, torch::kFloat32).clone().to(device_);
        training_loss = _train(tensor_sequence, actions_tensor, next_tensor_sequence, rewards_tensor, dones_tensor);
    }
    if (total_updates % training_loss_log_freq == 0) {
        cout << "ѕотери агента: \n" << training_loss << "\n"
            << "ќбщее кол-во наблюдений: " << total_observations << "\n";
    }
    if (total_updates % save_frequency == 0) {
        save();
    }
    total_updates++;
}
void ResourceWeightingDqnAgent::add_trajectory(
    const StateList& state,
    const ActionList& action,
    const StateList& next_state,
    const RewardList& reward,
    const DoneList& done
) {
    RewardList new_reward;
    for (const auto& el : reward) {
        new_reward.push_back(el * reward_scaling);
    }
    SampleRW sample = { state, action, next_state, new_reward, done };
    replay_buffer.push_back(sample);
    if (replay_buffer.size() > replay_buffer_length) replay_buffer.pop_front();
    total_observations++;
    if (total_observations >= initial_training_replay_size && total_observations % training_freq == 0)
        train();
}
void ResourceWeightingDqnAgent::resource_allocation_obs(
    const ResourceAllocationState& agent_state,
    const TaskWeightMap& actions,
    const ResourceAllocationState& next_agent_state,
    const TaskList& finished_tasks
) {
    if (agent_state.allocated_tasks.size() != actions.size()) {
        cout << "–азмер списка выполн€ющихс€ задач не совпадает с размером списка действий!\n";
        assert(false);
    }
    for (const auto& task_weight : actions) {
        if (
            find(agent_state.allocated_tasks.begin(), agent_state.allocated_tasks.end(), task_weight.first)
            == agent_state.allocated_tasks.end()
        ) {
            cout << "¬ списке выделенных задач не найдена задача из списка действий!\n";
            assert(false);
        }
    }
    for (const auto& task : finished_tasks) {
        if (task.get_stage() != TaskStage::COMPLETED && task.get_stage() != TaskStage::FAILED) {
            cout << "—тади€ задачи в списке завершенных задач не соответствует завершенному состо€нию!\n";
            assert(false);
        }
    }
    for (const auto& task : agent_state.allocated_tasks) {
        if (
            find(next_agent_state.allocated_tasks.begin(), next_agent_state.allocated_tasks.end(), task)
            == next_agent_state.allocated_tasks.end()
            &&
            find(finished_tasks.begin(), finished_tasks.end(), task)
            == finished_tasks.end()
        ) {
            cout << "«адача из списка выполн€ющихс€ задач текущего состо€ни€ не найдена в списке "
                "выделенных задач следующего состо€ни€ и в списке завершенных задач!\n";
            assert(false);
        }
    }
    if (agent_state.allocated_tasks.size() <= 1 || next_agent_state.allocated_tasks.size() <= 1) return;
    StateList obs = network_obs(agent_state.allocated_tasks, agent_state.server, agent_state.time_step);
    StateList next_obs = network_obs(next_agent_state.allocated_tasks, next_agent_state.server, next_agent_state.time_step);
    RewardList rewards;
    DoneList dones;
    ActionList actionlist;
    int k = 0;
    for (const auto& task_weight : actions) {
        const Task& task = task_weight.first;
        float action = task_weight.second.get();
        actionlist.push_back(action);
        float reward = 0.0f;
        for (const auto& finished_task : finished_tasks) {
            if (task != finished_task) {
                reward += (finished_task.get_stage() == TaskStage::COMPLETED ? success_reward : failed_reward) * other_task_discount;
            }
        }
        int i = 0;
        for (; i < next_agent_state.allocated_tasks.size(); i++) {
            if (task == next_agent_state.allocated_tasks[i]) break;
        }
        if (i < next_agent_state.allocated_tasks.size()) {
            dones.push_back(false);
        }
        else {
            next_obs.insert(next_obs.begin() + k, State(network_obs_width, 0.0f));
            auto finished_task_it = find(finished_tasks.begin(), finished_tasks.end(), task);
            const Task& finished_task = *finished_task_it;
            reward += (finished_task.get_stage() == TaskStage::COMPLETED ? success_reward : failed_reward);
            dones.push_back(true);
        }
        rewards.push_back(reward);
        k++;
    }
    add_trajectory(obs, actionlist, next_obs, rewards, dones);
}
TaskWeightMap ResourceWeightingDqnAgent::get_actions(
    const TaskList& allocated_tasks,
    const Server& server,
    int time_step,
    bool training
) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    ActionList actionlist;
    StateList obs;
    TaskWeightMap actions;
    if (training) {
        update_epsilon();
        bool get_action_tensor = true;
        for (int i = 0; i < allocated_tasks.size(); i++) {
            if (dis(gen) < epsilon) {
                actions[allocated_tasks[i]] = (rand() % num_actions) + 1.0f;
            }
            else {
                if (get_action_tensor) {
                    obs = network_obs(allocated_tasks, server, time_step);
                    vector<torch::Tensor> tensor_sequence;
                    for (const auto& state_obs : obs) {
                        vector<float> state = state_obs;
                        torch::Tensor state_tensor = torch::from_blob(state.data(), { static_cast<long>(state.size()) }, torch::kFloat32).clone();
                        tensor_sequence.push_back(state_tensor);
                    }
                    torch::Tensor obs_tensor = torch::stack(tensor_sequence, 0);
                    obs_tensor = obs_tensor.unsqueeze(0).to(device_);
                    torch::Tensor q_values = model_network->forward(obs_tensor).to(cpu_);
                    torch::Tensor action = torch::argmax(q_values, 2, false);
                    for (int j = 0; j < action.size(1); j++) {
                        actionlist.push_back(action[0][j].item<float>() + 1.0f);
                    }
                    get_action_tensor = false;
                }
                actions[allocated_tasks[i]] = actionlist[i];
            }
        }
    }
    else {
        obs = network_obs(allocated_tasks, server, time_step);
        vector<torch::Tensor> tensor_sequence;
        for (const auto& state_obs : obs) {
            vector<float> state = state_obs;
            torch::Tensor state_tensor = torch::from_blob(state.data(), { static_cast<long>(state.size()) }, torch::kFloat32).clone();
            tensor_sequence.push_back(state_tensor);
        }
        torch::Tensor obs_tensor = torch::stack(tensor_sequence, 0);
        obs_tensor = obs_tensor.unsqueeze(0).to(device_);
        torch::Tensor q_values = model_network->forward(obs_tensor).to(cpu_);
        torch::Tensor action = torch::argmax(q_values, 2, false);
        for (int j = 0; j < action.size(1); j++) {
            actionlist.push_back(action[0][j].item<float>() + 1.0f);
        }
        for (int i = 0; i < allocated_tasks.size(); ++i) {
            actions[allocated_tasks[i]] = actionlist[i];
        }
    }
    return actions;
}
