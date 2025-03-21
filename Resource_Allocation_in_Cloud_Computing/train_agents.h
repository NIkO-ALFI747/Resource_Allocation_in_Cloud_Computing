#pragma once
#include<random>
#include <chrono>
#include <ctime>
#include <torch/torch.h>
#include <fstream>
#include <filesystem>
#include"dqn_agent.h"
#include"env.h"
#include"resource_weighting_agent.h"
#include"task_pricing_agent.h"
#include "eval_results.h"
#include <optional>

typedef vector<TaskPricingDqnAgent> TaskPricingAgentList;
typedef vector<ResourceWeightingDqnAgent> ResourceWeightingAgentList;
typedef unordered_map<Server, TaskPricingDqnAgent*, ServerHash> TaskPricingAgentMap;
typedef unordered_map<Server, ResourceWeightingDqnAgent*, ServerHash> ResourceWeightingAgentMap;
typedef unordered_map<Server, optional<tuple<TaskPricingState, EnvAction, bool>>, ServerHash> ServerAuctionSubSampleMap;
typedef vector<tuple<TaskPricingState, EnvAction, TaskPricingState>> AuctionSubSampleList;

pair<TaskPricingAgentMap, ResourceWeightingAgentMap> allocate_agents(
    const EnvState& state,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list
);
EvalResults eval_agent(
    const vector<string>& env_filenames,
    int episode,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list
);
void train_agent(
    ResourceAllocationEnvironment& training_env,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list
);
void run_training(
    ResourceAllocationEnvironment& training_env,
    const vector<string>& eval_envs,
    int total_episodes,
    TaskPricingAgentList& task_pricing_agent_list,
    ResourceWeightingAgentList& resource_weighting_agent_list,
    int eval_frequency
);
vector<string> generate_eval_envs(
    ResourceAllocationEnvironment& eval_env,
    int num_evals,
    const string& folder,
    bool overwrite = false
);