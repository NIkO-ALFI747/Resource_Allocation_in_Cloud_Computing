#pragma once
#include "server.h"
#include <cassert>
#include <string>
#include <algorithm>

/// Server

Server::Server(
    int id,
    const string& name,
    float storage_cap,
    float computational_cap,
    float bandwidth_cap
) :
    id(id),
    name(name),
    storage_cap(storage_cap),
    computational_cap(computational_cap),
    bandwidth_cap(bandwidth_cap)
{}

TaskResourceUsageMap Server::allocate_bandwidth_resources(
    TaskWeightMap& loading_weights, TaskWeightMap& sending_weights,
    float available_storage, float available_bandwidth, int time_step
) const {
    TaskResourceUsageMap server_resource_usage;
    for (const auto& task_weight : loading_weights) {
        if (task_weight.first.get_stage() != TaskStage::LOADING) {
            cout << "Задача в словаре с весом выделяемых ресурсов загрузки должна находиться на этапе загрузки!\n";
            assert(false);
        }
    }
    for (const auto& task_weight : sending_weights) {
        if (task_weight.first.get_stage() != TaskStage::SENDING) {
            cout << "Задача в словаре с весом выделяемых ресурсов отправки должна находиться на этапе отправки!\n";
            assert(false);
        }
    }
    float bandwidth_total_weights = 0.0f, loading_total_weights = 0.0f, bandwidth_unit, storage_unit;
    float sending_resources_left, loading_resources_left;
    float sending_resources, loading_resources;
    TaskList tasks_been_updated;
    bool tasks_been_updated_ext;
    Task task;
    Weight weight;
    for (const auto& task_weight : loading_weights) {
        bandwidth_total_weights += task_weight.second.get();
        loading_total_weights += task_weight.second.get();
    }
    for (const auto& task_weight : sending_weights)
        bandwidth_total_weights += task_weight.second.get();
    
    tasks_been_updated_ext = true;
    while (tasks_been_updated_ext && (!loading_weights.empty() || !sending_weights.empty())) {
        tasks_been_updated_ext = false;
        tasks_been_updated.push_back(Task());
        while (!tasks_been_updated.empty() && (!loading_weights.empty() || !sending_weights.empty())) {
            tasks_been_updated.clear();
            bandwidth_unit = available_bandwidth / bandwidth_total_weights;
            for (const auto& task_weight : sending_weights) {
                task = task_weight.first;
                weight = task_weight.second;
                sending_resources_left = task.get_required_result_data() - task.get_sending_progress();
                if (sending_resources_left <= weight.get() * bandwidth_unit) {
                    task.allocate_sending_resources(sending_resources_left, time_step);
                    if (task.get_stage() != TaskStage::COMPLETED && task.get_stage() != TaskStage::FAILED) {
                        cout << "После выделения всех требуемых ресурсов отправки задача все еще имеет не завершенное состояние!\n";
                        assert(false);
                    }
                    server_resource_usage[task] = make_tuple(sending_resources_left, 0.0f, sending_resources_left);
                    tasks_been_updated.push_back(task_weight.first);
                    available_bandwidth -= sending_resources_left;
                    bandwidth_total_weights -= weight.get();
                }
            }
            for (const auto& task : tasks_been_updated) {
                sending_weights.erase(task);
            }
            for (const auto& task_weight : loading_weights) {
                storage_unit = available_storage / loading_total_weights;
                bandwidth_unit = available_bandwidth / bandwidth_total_weights;
                task = task_weight.first;
                weight = task_weight.second;
                loading_resources_left = task.get_required_storage() - task.get_loading_progress();
                if (loading_resources_left <= min(weight.get() * bandwidth_unit, (weight.get() * storage_unit) - task.get_required_result_data())) {
                    task.allocate_loading_resources(loading_resources_left, time_step);
                    if (task.get_stage() != TaskStage::COMPUTING && task.get_stage() != TaskStage::FAILED) {
                        cout << "После выделения всех требуемых ресурсов загрузки задача все еще не имеет этапа вычислений или неудачного завершения!\n";
                        assert(false);
                    }
                    server_resource_usage[task] = make_tuple(task.get_required_storage() + task.get_required_result_data(), 0.0f, loading_resources_left);
                    tasks_been_updated.push_back(task_weight.first);
                    available_storage -= (loading_resources_left + task.get_required_result_data());
                    available_bandwidth -= loading_resources_left;
                    bandwidth_total_weights -= weight.get();
                    loading_total_weights -= weight.get();
                }
            }
            for (const auto& task : tasks_been_updated) {
                loading_weights.erase(task);
            }
        }
        if (!loading_weights.empty()) {
            const auto& loading_task_weight = *loading_weights.begin();
            storage_unit = available_storage / loading_total_weights;
            bandwidth_unit = available_bandwidth / bandwidth_total_weights;
            task = loading_task_weight.first;
            weight = loading_task_weight.second;
            loading_resources_left = task.get_required_storage() - task.get_loading_progress();
            if (loading_resources_left <= weight.get() * bandwidth_unit) {
                if (weight.get() < 1.0f) {
                    loading_resources = min(weight.get() * storage_unit, weight.get() * loading_resources_left);
                }
                else {
                    loading_resources = min(storage_unit, loading_resources_left / 2.0f);
                }
            }
            else {
                loading_resources = min(weight.get() * bandwidth_unit, weight.get() * storage_unit);
            }
            if (loading_resources >= loading_resources_left - 0.0001f) {
                loading_resources = 0;
            }
            task.allocate_loading_resources(loading_resources, time_step);
            if (task.get_stage() != TaskStage::LOADING && task.get_stage() != TaskStage::FAILED) {
                cout << "После выделения некоторых ресурсов загрузки задача не находится в состоянии загрузки или неудачного завершения!\n";
                assert(false);
            }
            server_resource_usage[task] = make_tuple(task.get_loading_progress(), 0.0f, loading_resources);
            tasks_been_updated_ext = true;
            available_storage -= loading_resources;
            available_bandwidth -= loading_resources;
            bandwidth_total_weights -= weight.get();
            loading_total_weights -= weight.get();
            loading_weights.erase(loading_task_weight.first);
        }
    }
    bandwidth_unit = available_bandwidth / bandwidth_total_weights;
    for (const auto& task_weight : sending_weights) {
        task = task_weight.first;
        weight = task_weight.second;
        sending_resources = weight.get() * bandwidth_unit;
        sending_resources_left = task.get_required_result_data() - task.get_sending_progress();
        task.allocate_sending_resources(sending_resources, time_step);
        server_resource_usage[task] = make_tuple(sending_resources_left, 0.0f, sending_resources);
    }
    return server_resource_usage;
}
TaskResourceUsageMap Server::allocate_computing_resources(
    TaskWeightMap& computing_weights, float available_computation, int time_step
) const {
    TaskResourceUsageMap server_resource_usage;
    for (const auto& task_weight : computing_weights) {
        if (task_weight.first.get_stage() != TaskStage::COMPUTING) {
            cout << "Задача в словаре с весом выделяемых вычислительных ресурсов должна находиться на этапе вычислений!\n";
            assert(false);
        }
    }
    float computing_total_weights, computing_unit, computing_resources_left;
    float computing_resources;
    TaskList tasks_been_updated = {Task()};
    Task task;
    Weight weight;
    computing_total_weights = 0.0f;
    for (const auto& task_weight : computing_weights)
        computing_total_weights += task_weight.second.get();

    while (!tasks_been_updated.empty() && !computing_weights.empty()) {
        tasks_been_updated.clear();
        for (const auto& task_weight : computing_weights) {
            computing_unit = available_computation / computing_total_weights;
            task = task_weight.first;
            weight = task_weight.second;
            computing_resources_left = task.get_required_computation() - task.get_computing_progress();
            if (computing_resources_left <= weight.get() * computing_unit) {
                task.allocate_computing_resources(computing_resources_left, time_step);
                if (task.get_stage() != TaskStage::SENDING && task.get_stage() != TaskStage::FAILED) {
                    cout << "После выделения всех требуемых ресурсов вычислений задача все еще не имеет этапа отправки или неудачного завершения!\n";
                    assert(false);
                }
                server_resource_usage[task] = make_tuple(task.get_required_storage() + task.get_required_result_data(), computing_resources_left, 0.0f);
                tasks_been_updated.push_back(task_weight.first);
                available_computation -= computing_resources_left;
                computing_total_weights -= weight.get();
            }
        }
        for (const auto& task : tasks_been_updated) {
            computing_weights.erase(task);
        }
    }
    if (!computing_weights.empty()) {
        computing_unit = available_computation / computing_total_weights;
        for (const auto& task_weight : computing_weights) {
            task = task_weight.first;
            weight = task_weight.second;
            computing_resources = computing_unit * weight.get();
            task.allocate_computing_resources(computing_resources, time_step);
            server_resource_usage[task] = make_tuple(task.get_required_storage() + task.get_required_result_data(), computing_resources, 0.0f);
        }
    }
    return server_resource_usage;
}
tuple<TaskList, TaskList> Server::allocate_resources(
    const TaskWeightMap& resource_weights, int time_step, float error
) const {
    assert_validity();
    if (resource_weights.size() == 0) {
        cout << "Словарь с весами выделяемых ресурсов для задач должен быть заполнен!\n";
        assert(false);
    }
    for (const auto& task_weight : resource_weights) {
        if (
            task_weight.first.get_stage() != TaskStage::LOADING &&
            task_weight.first.get_stage() != TaskStage::COMPUTING &&
            task_weight.first.get_stage() != TaskStage::SENDING
        ) {
            cout << "Задачи в словаре с весами выделяемых ресурсов должны находиться на стадии загрузки, вычислений либо отправки!\n";
            assert(false);
        }
        task_weight.first.assert_validity();
    }
    Task task;
    Weight weight;
    float available_storage = storage_cap, available_storage2 = storage_cap;
    float available_computation = computational_cap;
    float available_bandwidth = bandwidth_cap;
    TaskResourceUsageMap computation_resource_usage, bandwidth_resource_usage, server_resource_usage;
    TaskWeightMap loading_weights, computing_weights, sending_weights;
    TaskList sending_task_map;

    for (const auto& task_weight : resource_weights) {
        task = task_weight.first;
        weight = task_weight.second;
        if (task.get_stage() == TaskStage::LOADING) {
            loading_weights[task] = weight;
            available_storage -= task.get_loading_progress();
        }
        else if (task.get_stage() == TaskStage::COMPUTING) {
            computing_weights[task] = weight;
            available_storage -= (task.get_loading_progress() + task.get_required_result_data());
        }
        else if (task.get_stage() == TaskStage::SENDING) {
            sending_weights[task] = weight;
            sending_task_map.push_back(task);
            available_storage -= (task.get_required_result_data() - task.get_sending_progress());
        }
    }
    if (available_storage < 0.0f) {
        if (available_storage >= -error) {
            available_storage = 0.0f;
        }
        else {
            cout << "Задачам было выделено больше ресурсов хранилища, чем доступно на сервере!\n";
            assert(false);
        }
    }
    if (computing_weights.size() > 0)
        computation_resource_usage = allocate_computing_resources(computing_weights, available_computation, time_step);
    if (loading_weights.size() > 0 || sending_weights.size() > 0)
        bandwidth_resource_usage = allocate_bandwidth_resources(loading_weights, sending_weights, available_storage, available_bandwidth, time_step);
    server_resource_usage.insert(computation_resource_usage.begin(), computation_resource_usage.end());
    server_resource_usage.insert(bandwidth_resource_usage.begin(), bandwidth_resource_usage.end());
    float storage_usage = 0.0f, computation_usage = 0.0f, bandwidth_usage = 0.0f;
    for (const auto& task_res_usage : server_resource_usage) {
        storage_usage += get<0>(task_res_usage.second.get_res_usage());
        computation_usage += get<1>(task_res_usage.second.get_res_usage());
        bandwidth_usage += get<2>(task_res_usage.second.get_res_usage());
    }
    if (storage_usage > storage_cap + error) {
        cout << "Доступно меньше ресурсов хранилища, чем выделено!\n";
        assert(false);
    }
    if (computation_usage > computational_cap + error) {
        cout << "Доступно меньше вычислительных ресурсов, чем выделено!\n";
        assert(false);
    }
    if (bandwidth_usage > bandwidth_cap + error) {
        cout << "Доступно меньше ресурсов пропускной способности, чем выделено!\n";
        assert(false);
    }
    TaskList unfinished_tasks, completed_tasks;
    for (const auto& task_res_usage : server_resource_usage) {
        task = task_res_usage.first;
        if (task.get_stage() != TaskStage::COMPLETED && task.get_stage() != TaskStage::FAILED) unfinished_tasks.push_back(task);
        else completed_tasks.push_back(task);
    }
    return make_tuple(unfinished_tasks, completed_tasks);
}
void Server::assert_validity() const {
    if (storage_cap <= 0.0f || computational_cap <= 0.0f || bandwidth_cap <= 0.0f) {
        cout << "Общий объем имеющихся ресурсов должен быть положительным!\n";
        assert(false);
    }
}

string Server::server_to_string() const {
    return (
        "Сервер " + name + ":\n"
        + "Идентификатор: " + to_string(id) + "\n"
        + "Общий объем хранилища: " + to_string(storage_cap) + "\n"
        + "Общий объем вычислительной мощности: " + to_string(computational_cap) + "\n"
        + "Общая пропускная способность сети сервера: " + to_string(bandwidth_cap) + "\n"
    );
}
string Server::serverlist_to_string(const ServerList& serverlist, bool detail_flag) {
    string serverlist_str = "";
    if (detail_flag) {
        for (const auto& server : serverlist) {
            serverlist_str += server.server_to_string() + "\n";
        }
    }
    else {
        for (const auto& server : serverlist) {
            serverlist_str += server.get_name() + ", ";
        }
        if (!serverlist_str.empty()) {
            serverlist_str = serverlist_str.substr(0, serverlist_str.size() - 2);
        }
    }
    return serverlist_str + "\n";
}
string Server::server_tasklist_map_to_string(const ServerTaskListMap& server_tasklist_map, bool detail_flag) {
    string server_tasklist_map_str = "";
    if (detail_flag) {
        for (const auto& server_tasklist : server_tasklist_map) {
            server_tasklist_map_str += "Key: \n" + server_tasklist.first.server_to_string() + "\n";
            server_tasklist_map_str += "Value: \n" + Task::tasklist_to_string(server_tasklist.second, detail_flag) + "\n";
        }
    }
    else {
        for (const auto& server_tasklist : server_tasklist_map) {
            server_tasklist_map_str += "Key: \n" + server_tasklist.first.get_name() + "\n";
            server_tasklist_map_str += "Value: \n" + Task::tasklist_to_string(server_tasklist.second, detail_flag) + "\n";
        }
    }
    
    return server_tasklist_map_str;
}
bool Server::operator==(const Server& server) const {
    return id == server.id;
}

int Server::get_id() const {
    return id;
}
string Server::get_name() const {
    return name;
}
float Server::get_storage_cap() const {
    return storage_cap;
};
float Server::get_computational_cap() const {
    return computational_cap;
};
float Server::get_bandwidth_cap() const {
    return bandwidth_cap;
};

/// ServerHash

int ServerHash::operator()(const Server& server) const {
    return hash<int>{}(server.get_id());
}