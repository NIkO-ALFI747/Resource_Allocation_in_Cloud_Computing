#pragma once
#include"train_agents.h"
#include"dq_network.h"
#include"dqn_alg.h"
#include <torch/torch.h>
#include <filesystem>

void run_dqn() {
    ResourceAllocationEnvironment env({ ".\\training\\settings\\basic.env" }); // can be the absolute path
    env.reset();
    cout << env.env_to_string()
        << "\nАукционная задача: \n" << env.state.auction_task.task_to_string() << "\n"
        << env.state.env_state_to_string() << "\n";
    vector<string> eval_envs = generate_eval_envs(env, 20, ".\\training\\settings\\eval_envs\\algo");
    cout << "Файлы сред для оценки агентов обучения:\n";
    for (string eval_env : eval_envs)
        cout << eval_env << "\n";
    string save_env_folder = "training\\models\\init\\";
    string save_path = filesystem::current_path().string() + ".\\" + save_env_folder;
    string rw_save_path = filesystem::current_path().string() + ".\\" + save_env_folder;
    TaskPricingAgentList task_pricing_agent_list;
    ResourceWeightingAgentList resource_weighting_agent_list;
    for (int agent_num = 0; agent_num < 1; ++agent_num) {
        torch::nn::Sequential network = create_lstm_dq_network(10, 31);
        torch::nn::Sequential rw_network = create_bidirectional_dq_network(10, 21);
        if (!filesystem::exists(".\\" + save_env_folder)) {
            filesystem::create_directories(save_env_folder);
        }
        save_path += "network" + to_string(agent_num) + ".pt";
        torch::save(network, save_path);
        rw_save_path += "rw_network" + to_string(agent_num) + ".pt";
        torch::save(rw_network, save_path);
        task_pricing_agent_list.push_back(TaskPricingDqnAgent(agent_num, network));
        resource_weighting_agent_list.push_back(ResourceWeightingDqnAgent(0, rw_network));
    }
    run_training(env, eval_envs, 80, task_pricing_agent_list, resource_weighting_agent_list, 2);
    for (auto agent : task_pricing_agent_list)
        agent.save();
    for (auto agent : resource_weighting_agent_list)
        agent.save();
}