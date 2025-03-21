#pragma once
#include "train_agents.h"

pair<TaskPricingAgentMap, ResourceWeightingAgentMap> allocate_agents(
    const EnvState& state,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list
) {
    TaskPricingAgentMap server_task_pricing_agent_map;
    ResourceWeightingAgentMap server_resource_weighting_agent_map;
    random_device rd;
    mt19937 gen(rd());
    for (const auto& server_tasklist : state.server_tasklist_map) {
        uniform_int_distribution<> task_pricing_agent_dist(0, task_pricing_agent_list.size() - 1);
        server_task_pricing_agent_map[server_tasklist.first] = &task_pricing_agent_list[task_pricing_agent_dist(gen)];
        uniform_int_distribution<> resource_weighting_agent_dist(0, resource_weighting_agent_list.size() - 1);
        server_resource_weighting_agent_map[server_tasklist.first] = &resource_weighting_agent_list[resource_weighting_agent_dist(gen)];
    }
    return make_pair(server_task_pricing_agent_map, server_resource_weighting_agent_map);
}
EvalResults eval_agent(
    const vector<string>& env_filenames,
    int episode,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list
) {
    EvalResults results;
    for (const auto& env_filename : env_filenames) {
        auto env_state_tuple = ResourceAllocationEnvironment::load_env(env_filename);
        //cout << "load_env\n";
        ResourceAllocationEnvironment eval_env = get<0>(env_state_tuple);
        EnvState state = get<1>(env_state_tuple);
        TaskPricingAgentMap server_task_pricing_agent_map;
        ResourceWeightingAgentMap server_resource_weighting_agent_map;
        tie(server_task_pricing_agent_map, server_resource_weighting_agent_map) = allocate_agents(state, task_pricing_agent_list, resource_weighting_agent_list);
        unordered_map<string, string> info;
        ServerRewardMap rewards;
        bool done = false;
        while (!done) {
            if (state.auction_task.get_id() != -1) {
                ServerActionMap bidding_actions;
                for (const auto& server_tasklist : state.server_tasklist_map) {
                    const Server& server = server_tasklist.first;
                    bidding_actions[server] = EnvAction(
                        (*server_task_pricing_agent_map[server]).bid(state.auction_task, server_tasklist.second, server, state.time_step)
                    );
                }
                tie(state, rewards, done, info) = eval_env.step(bidding_actions);
                results.auction(bidding_actions, rewards);
            }
            else {
                ServerActionMap weighting_actions;
                for (const auto& server_tasklist : state.server_tasklist_map) {
                    const Server& server = server_tasklist.first;
                    weighting_actions[server] = EnvAction(
                        (*server_resource_weighting_agent_map[server]).weight(server_tasklist.second, server, state.time_step)
                    );
                }
                tie(state, rewards, done, info) = eval_env.step(weighting_actions);
                results.resource_allocation(weighting_actions, rewards);
            }
        }
        results.finished_env();
    }
    results.save(episode);
    return results;
}
void train_agent(
    ResourceAllocationEnvironment& training_env,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list
) {
    EnvState state = training_env.reset();
    EnvState next_state;
    TaskPricingAgentMap server_task_pricing_agent_map;
    ResourceWeightingAgentMap server_resource_weighting_agent_map;
    tie(server_task_pricing_agent_map, server_resource_weighting_agent_map) = allocate_agents(state, task_pricing_agent_list, resource_weighting_agent_list);
    ServerAuctionSubSampleMap server_auction_subsample_map;
    AuctionSubSampleList successful_auction_subsample_list;
    for (const auto& server_tasklist : state.server_tasklist_map)
        server_auction_subsample_map[server_tasklist.first] = nullopt;

    bool done = false;
    while (!done) {
        if (state.auction_task.get_id() != -1) {
            ServerActionMap auction_prices;
            for (const auto& server_tasklist : state.server_tasklist_map) {
                const Server& server = server_tasklist.first;
                auction_prices[server] = EnvAction(
                    (*server_task_pricing_agent_map[server]).bid(state.auction_task, server_tasklist.second, server, state.time_step, true)
                );
            }
            auto step_result = training_env.step(auction_prices);
            next_state = get<0>(step_result);
            auto& rewards = get<1>(step_result);
            done = get<2>(step_result);
            auto& info = get<3>(step_result);
            for (const auto& server_tasklist : state.server_tasklist_map) {
                TaskPricingState current_state(state.auction_task, server_tasklist.second, server_tasklist.first, state.time_step);
                if (server_auction_subsample_map[server_tasklist.first].has_value()) {
                    const auto& [previous_state, previous_action, is_previous_auction_win] = server_auction_subsample_map[server_tasklist.first].value();
                    if (is_previous_auction_win)
                        successful_auction_subsample_list.emplace_back(previous_state, previous_action, current_state);
                    else
                        (*server_task_pricing_agent_map[server_tasklist.first]).failed_auction_bid(previous_state, previous_action.action.get(), current_state);
                }
                server_auction_subsample_map[server_tasklist.first] = make_tuple(current_state, auction_prices[server_tasklist.first], rewards.find(server_tasklist.first) != rewards.end());
            }
        }
        else {
            ServerActionMap weighting_actions;
            for (const auto& server_tasklist : state.server_tasklist_map) {
                const Server& server = server_tasklist.first;
                weighting_actions[server] = EnvAction(
                    (*server_resource_weighting_agent_map[server]).weight(server_tasklist.second, server, state.time_step, true)
                );
            }
            auto step_result = training_env.step(weighting_actions);
            next_state = get<0>(step_result);
            auto& finished_server_tasklist_map = get<1>(step_result);
            done = get<2>(step_result);
            auto& info = get<3>(step_result);
            for (const auto& server_tasklist : finished_server_tasklist_map) {
                for (const auto& finished_task : server_tasklist.second.tasklist) {
                    auto successful_auction = find_if(
                        successful_auction_subsample_list.begin(),
                        successful_auction_subsample_list.end(),
                        [&finished_task](const auto& auction_agent_subsample) {
                            auto auction_state = get<0>(auction_agent_subsample);
                            return auction_state.auction_task == finished_task;
                        }
                    );
                    if (successful_auction != successful_auction_subsample_list.end()) {
                        const auto& [auction_state, price, next_auction_state] = *successful_auction;
                        successful_auction_subsample_list.erase(successful_auction);
                        (*server_task_pricing_agent_map[server_tasklist.first]).winning_auction_bid(auction_state, price.action.get(), finished_task, next_auction_state);
                    }
                    else {
                        cout << "В списке подтраекторий проведенных успешных аукционов не найдена задача из списка завершенных задач!\n";
                        assert(false);
                    }
                }
            }
            for (const auto& server_tasklist : state.server_tasklist_map) {
                const Server& server = server_tasklist.first;
                ResourceAllocationState agent_state(server_tasklist.second, server, state.time_step);
                ResourceAllocationState next_agent_state(next_state.server_tasklist_map[server], server, next_state.time_step);
                (*server_resource_weighting_agent_map[server]).resource_allocation_obs(
                    agent_state, weighting_actions[server].task_action_map, next_agent_state, finished_server_tasklist_map[server].tasklist
                );
            }
        }
        for (const auto& server_tasklist : next_state.server_tasklist_map) {
            for (const auto& task : server_tasklist.second) {
                if (task.get_auction_time() > next_state.time_step || next_state.time_step > task.get_deadline()) {
                    cout << "Текущий временной шаг следующего состояния меньше времени проведения аукциона или больше либо равен крайнему сроку завершения задачи!\n";
                    assert(false);
                }
            }
        }
        state = next_state;
    }
}
void run_training(
    ResourceAllocationEnvironment& training_env,
    const vector<string>& eval_envs,
    int total_episodes,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list,
    int eval_frequency
) {
    for (int episode = 0; episode < total_episodes; ++episode) {
        if (episode % 5 == 0) {
            //time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
            cout << "\nЭпизод: " << episode /* << " в " << ctime(&now) */ << endl;
        }
        train_agent(training_env, task_pricing_agent_list, resource_weighting_agent_list);
        if (episode % eval_frequency == 0) {
            eval_agent(eval_envs, episode, task_pricing_agent_list, resource_weighting_agent_list);
        }
    }
}
vector<string> generate_eval_envs(ResourceAllocationEnvironment& eval_env, int num_evals,
    const string& folder,bool overwrite) {
    vector<string> eval_files;
    if (!filesystem::exists(folder)) {
        filesystem::create_directories(folder);
    }
    for (int eval_num = 0; eval_num < num_evals; ++eval_num) {
        string eval_file = folder + "\\eval_" + to_string(eval_num) + ".env";
        eval_files.push_back(eval_file);
        if (overwrite || !filesystem::exists(eval_file)) {
            eval_env.reset();
            eval_env.save_env(eval_file);
        }
    }
    return eval_files;
}
