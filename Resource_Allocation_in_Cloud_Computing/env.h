#pragma once
#include "env_state.h"

class ResourceAllocationEnvironment {
private:
    vector<string> env_settings;
    string env_name;
    int total_time_steps = -1;
public:
    TaskList unallocated_tasks;
    EnvState state;

    static void assert_total_time_steps_validity(int total_time_steps);
    static void assert_server_tasklist_map_validity(const ServerTaskListMap& server_tasklist_map);
    static void assert_un_tasks_time_step_validity(const TaskList& unallocated_tasks, int time_step = 0);
    static void assert_unallocated_tasks_validity(const TaskList& unallocated_tasks);

    ResourceAllocationEnvironment(
        const vector<string>& env_settings
    );
    ResourceAllocationEnvironment(
        const string& env_name,
        const ServerTaskListMap& server_tasklist_map,
        const TaskList& unallocated_tasks,
        int total_time_steps,
        int time_step = 0
    );

    Task next_auction_task(int time_step);
    void save_env(const string& filename);
    static tuple<ResourceAllocationEnvironment, EnvState> load_env(const string& filename);
    tuple<string, ServerList, TaskList, int> load_settings(const string& filename);
    EnvState reset();
    tuple<EnvState, ServerRewardMap, bool, unordered_map<string, string>> step(const ServerActionMap& actions);
    string env_to_string() const;
};
