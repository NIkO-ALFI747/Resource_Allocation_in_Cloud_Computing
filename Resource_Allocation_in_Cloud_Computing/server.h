#pragma once
#include "task.h"
class Server;
struct ServerHash {
    int operator()(const Server& server) const;
};

typedef vector<Server> ServerList;
typedef unordered_map<Server, TaskList, ServerHash> ServerTaskListMap;

class Server {
private:
    int id;
    string name;
    float storage_cap;
    float computational_cap;
    float bandwidth_cap;
public:
    Server(
        int id = -1,
        const string& name = "unnamed",
        float storage_cap = -1.0f,
        float computational_cap = -1.0f,
        float bandwidth_cap = -1.0f
    );

    TaskResourceUsageMap allocate_bandwidth_resources(
        TaskWeightMap& loading_weights, TaskWeightMap& sending_weights,
        float available_storage, float available_bandwidth, int time_step
    ) const;
    TaskResourceUsageMap allocate_computing_resources(
        TaskWeightMap& computing_weights, float available_computation, int time_step
    ) const;
    tuple<TaskList, TaskList> allocate_resources(
        const TaskWeightMap& resource_weights, int time_step, float error = 0.1f
    ) const;
    void assert_validity() const;

    string server_to_string() const;
    static string serverlist_to_string(const ServerList& serverlist, bool detail_flag = false);
    static string server_tasklist_map_to_string(const ServerTaskListMap& server_tasklist_map, 
        bool detail_flag = false);
    bool operator==(const Server& server) const;

    int get_id() const;
    string get_name() const;
    float get_storage_cap() const;
    float get_computational_cap() const;
    float get_bandwidth_cap() const;
};
