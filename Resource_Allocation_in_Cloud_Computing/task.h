#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include "task_stage.h"
class Task;
class ResourceUsage;
class Weight;
struct TaskHash {
    int operator()(const Task& task) const;
};

typedef vector<Task> TaskList;
typedef unordered_map<Task, Weight, TaskHash> TaskWeightMap;
typedef tuple<float, float, float> ResourceUsageType;
typedef unordered_map<Task, ResourceUsage, TaskHash> TaskResourceUsageMap;

class Task {
private:
    int id;
    string name;
    int auction_time;
    int deadline;
    
    float required_storage;
    float required_computation;
    float required_result_data;
    
    float loading_progress;
    float computing_progress;
    float sending_progress;
    
    TaskStage stage;
    float price;
public:
    Task(
        int id = -1,
        const string& name = "unnamed",
        int auction_time = -1,
        int deadline = -1,
        float required_storage = -1.0f,
        float required_computation = -1.0f,
        float required_result_data = -1.0f,

        float loading_progress = 0.0f,
        float computing_progress = 0.0f,
        float sending_progress = 0.0f,
        TaskStage stage = TaskStage::UNASSIGNED,
        float price = -1.0f
    );

    void assign_server(float price, int time_step);
    TaskStage has_failed(const TaskStage& updated_stage, int time_step) const;
    void allocate_loading_resources(float loading_resources, int time_step);
    void allocate_computing_resources(float computing_resources, int time_step);
    void allocate_sending_resources(float sending_resources, int time_step);
    void assert_validity() const;

    string task_to_string() const;
    static string tasklist_to_string(const TaskList& tasklist, bool detail_flag = false);
    static string task_weight_map_to_string(const TaskWeightMap& task_weight_map);
    static string task_resorce_usage_map_to_string(const TaskResourceUsageMap& task_res_usage_map);
    bool deeply_equal(const Task& task) const;
    bool operator==(const Task& task) const;
    bool operator!=(const Task& task) const;

    int get_id() const;
    string get_name() const;
    int get_auction_time() const;
    int get_deadline() const;
    float get_required_storage() const;
    float get_required_computation() const;
    float get_required_result_data() const;
    float get_loading_progress() const;
    float get_computing_progress() const;
    float get_sending_progress() const;
    TaskStage get_stage() const;
    float get_price() const;
};

class ResourceUsage {
private:
    ResourceUsageType res_usage;
public:
    static void assert_validity(const ResourceUsageType& res_usage);
    ResourceUsage& operator=(const ResourceUsageType& res_usage);
    ResourceUsage();
    ResourceUsage(const ResourceUsageType& res_usage);
    string res_usage_to_string() const;
    ResourceUsageType get_res_usage() const;
};
class Weight {
private:
    float weight = -1.0;
public:
    static void assert_validity(float weight);
    Weight& operator=(const float weight);
    Weight() {};
    Weight(float weight);
    float get() const;
};
class Price {
private:
    float price = -1.0;
public:
    static void assert_validity(float price);
    Price& operator=(const float price);
    Price() {};
    Price(float price);
    float get() const;
};