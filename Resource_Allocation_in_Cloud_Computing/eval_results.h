#pragma once
#include "env.h"

class EvalResults {
public:
    float total_winning_prices = 0.0f;
    vector<float> winning_prices;
    int num_auctions = 0;
    vector<float> auction_actions;
    int num_completed_tasks = 0;
    int num_failed_tasks = 0;
    float total_prices = 0.0f;
    int num_resource_allocations = 0;
    vector<float> weighting_actions;
    vector<int> env_attempted_tasks = { 0 };
    vector<int> env_completed_tasks = { 0 };
    vector<int> env_failed_tasks = { 0 };

    void auction(const ServerActionMap& actions, const ServerRewardMap& rewards);
    void resource_allocation(const ServerActionMap& actions, const ServerRewardMap& rewards);
    void finished_env();
    void save(int episode);
};
