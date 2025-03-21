#pragma once
#include "server.h"
class EnvAction;
class EnvReward;

typedef unordered_map<Server, EnvAction, ServerHash> ServerActionMap;
typedef unordered_map<Server, EnvReward, ServerHash> ServerRewardMap;

class EnvState {
public:
    ServerTaskListMap server_tasklist_map;
    Task auction_task;
    int time_step;
    EnvState(
        const ServerTaskListMap& server_tasklist_map = {},
        const Task auction_task = Task(),
        const int time_step = 0
    );
    string env_state_to_string() const;
};
class EnvReward {
public:
    float reward;
    TaskList tasklist;
    EnvReward();
    EnvReward(float reward);
    EnvReward(const TaskList& tasklist);
    size_t size() const;
};
class EnvAction {
public:
    Price action;
    TaskWeightMap task_action_map;
    EnvAction();
    EnvAction(float action);
    EnvAction(const TaskWeightMap& task_action_map);
    size_t size() const;
};