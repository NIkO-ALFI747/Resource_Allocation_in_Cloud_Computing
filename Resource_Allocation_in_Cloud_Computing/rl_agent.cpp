#pragma once
#include "rl_agent.h"
#include <fstream>
#include <string>
#include <cassert>
#include <random>
#include <deque>
#include <algorithm>
#include <iterator>
#include <torch/torch.h>

torch::Tensor error_loss_func(torch::Tensor target, torch::Tensor actual) {
    torch::nn::SmoothL1Loss criterion;
    torch::Tensor loss = criterion(target, actual);
    return loss;
}

ReinforcementLearningAgent::ReinforcementLearningAgent(
    int batch_size,
    LossFunc error_loss_fn,
    int initial_training_replay_size,
    int training_freq,
    float discount_factor,
    int replay_buffer_length,
    int save_frequency,
    string save_folder,
    int training_loss_log_freq,
    float reward_scaling
) :
    batch_size(batch_size),
    error_loss_fn(error_loss_func),
    initial_training_replay_size(initial_training_replay_size),
    training_freq(training_freq),
    training_loss_log_freq(training_loss_log_freq),
    reward_scaling(reward_scaling),
    discount_factor(discount_factor),
    total_updates(0),
    total_actions(0),
    replay_buffer_length(replay_buffer_length),
    total_observations(0),
    save_frequency(save_frequency),
    save_folder(save_folder)
{
    if (batch_size <= 0) {
        cout << "–азмер буфера воспроизведени€ опыта должен быть положительным!\n";
        assert(false);
    }
    if (training_freq <= 0 || save_frequency <= 0 || training_loss_log_freq <= 0) {
        cout << "„астота обновлени€ нейронной сети, сохранени€ модели и фиксировани€ потерь должны быть положительными!\n";
        assert(false);
    }
    if (initial_training_replay_size <= 0 || replay_buffer_length <= 0) {
        cout << "Ќачальный (дл€ начала обучени€) и общий размеры буфера воспроизведени€ опыта должны быть положительными!\n";
        assert(false);
    }
}
State ReinforcementLearningAgent::normalise_task(const Task& task, const Server& server, int time_step) {
    return {
        task.get_required_storage() / server.get_storage_cap(),
        task.get_required_storage() / server.get_bandwidth_cap(),
        task.get_required_computation() / server.get_computational_cap(),
        task.get_required_result_data() / server.get_storage_cap(),
        task.get_required_result_data() / server.get_bandwidth_cap(),
        static_cast<float>(task.get_deadline() - time_step),
        task.get_loading_progress(),
        task.get_computing_progress(),
        task.get_sending_progress()
    };
}
void ReinforcementLearningAgent::update_target_network(torch::nn::Sequential& model_network, torch::nn::Sequential& target_network, float tau) { 
    torch::autograd::GradMode::set_enabled(false);
    if (model_network->parameters().size() != target_network->parameters().size()) {
        cout << "–азмер целевой сети отличаетс€ от размера сети модели!\n";
        assert(false);
    }
    for (int i = 0; i < model_network->parameters().size(); i++) {
        torch::Tensor updated_param = torch::tensor(1 - tau) * target_network->parameters()[i].data() + torch::tensor(tau) * model_network->parameters()[i].data();
        target_network->parameters()[i].set_data(updated_param);
    }
    torch::autograd::GradMode::set_enabled(true);
}
