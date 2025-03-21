#pragma once
#include "env.h"
#include "json.hpp"
#include <string>
#include <fstream>
using json = nlohmann::json;

void ResourceAllocationEnvironment::assert_total_time_steps_validity(int total_time_steps) {
    if (total_time_steps <= 0) {
        cout << "Общее кол-во временных шагов в среде должно быть положительным!\n";
        assert(false);
    }
}
void ResourceAllocationEnvironment::assert_server_tasklist_map_validity(const ServerTaskListMap& server_tasklist_map) {
    if (server_tasklist_map.empty()) {
        cout << "Словарь серверов и выполняющихся на этих серверах задач не задан!\n";
        assert(false);
    }
    for (const auto& server_tasklist : server_tasklist_map) {
        server_tasklist.first.assert_validity();
        for (auto task : server_tasklist.second) {
            if (task.get_stage() == TaskStage::UNASSIGNED || task.get_stage() == TaskStage::COMPLETED || task.get_stage() == TaskStage::FAILED) {
                cout << "Стадия задачи в словаре серверов и выполняющихся на них задач не соответствует стадии выполнения!\n";
                assert(false);
            }
            task.assert_validity();
        }
    }
}
void ResourceAllocationEnvironment::assert_un_tasks_time_step_validity(const TaskList& unallocated_tasks, int time_step) {
    if (time_step < 0) {
        cout << "Текущий временной шаг среды должен быть положительным!\n";
        assert(false);
    }
    if (!unallocated_tasks.empty())
    {
        if (time_step > unallocated_tasks[0].get_auction_time()) {
            cout << "Текущий входной временной шаг превышает время проведения аукциона для первой задачи в списке нераспределенных задач\n";
            assert(false);
        }
    }
}
void ResourceAllocationEnvironment::assert_unallocated_tasks_validity(const TaskList& unallocated_tasks) {
    if (!unallocated_tasks.empty())
    {
        for (int i = 0; i < unallocated_tasks.size() - 1; i++) {
            if (unallocated_tasks[i].get_auction_time() > unallocated_tasks[i + 1].get_auction_time()) {
                cout << "В списке нераспределенных задач, задачи по времени аукциона расположены не по возрастанию!\n";
                assert(false);
            }
        }
        for (auto task : unallocated_tasks) {
            if (task.get_stage() != TaskStage::UNASSIGNED) {
                cout << "Стадия задачи в списке нераспределенных задач не соответствует стадии \"не назначена\"!\n";
                assert(false);
            }
            task.assert_validity();
        }
    }
}

ResourceAllocationEnvironment::ResourceAllocationEnvironment
(
    const vector<string>& env_settings
) {
    if (!env_settings.empty()) {
        this->env_settings = env_settings;
        this->env_name = "unnamed";
        this->total_time_steps = -1;
        this->unallocated_tasks = {};
        state = EnvState();
    }
    else {
        cout << "Список настроек среды должен быть задан!\n";
        assert(false);
    }
}
ResourceAllocationEnvironment::ResourceAllocationEnvironment
(
    const string& env_name,
    const ServerTaskListMap& server_tasklist_map,
    const TaskList& unallocated_tasks,
    int total_time_steps,
    int time_step
) {
    if (!this->env_settings.empty()) this->env_settings.clear();
    this->env_name = env_name;
    assert_total_time_steps_validity(total_time_steps);
    this->total_time_steps = total_time_steps;
    if (!unallocated_tasks.empty())
    {
        assert_unallocated_tasks_validity(unallocated_tasks);
        assert_un_tasks_time_step_validity(unallocated_tasks, time_step);
        this->unallocated_tasks = unallocated_tasks;
        Task auction_task;
        if (this->unallocated_tasks[0].get_auction_time() == time_step){
            auction_task = this->unallocated_tasks[0];
            this->unallocated_tasks.erase(this->unallocated_tasks.begin());
        }
        else {
            auction_task = Task();
        }
        assert_server_tasklist_map_validity(server_tasklist_map);
        state = EnvState(server_tasklist_map, auction_task, time_step);
    }
    else {
        this->unallocated_tasks = {};
        state = {};
    }
}

Task ResourceAllocationEnvironment::next_auction_task(int time_step) {
    assert_un_tasks_time_step_validity(unallocated_tasks, time_step);
    if (!unallocated_tasks.empty()) {
        if (time_step == unallocated_tasks[0].get_auction_time()) {
            Task task = unallocated_tasks[0];
            unallocated_tasks.erase(unallocated_tasks.begin());
            return task;
        }
    }
    return Task();
}
void ResourceAllocationEnvironment::save_env(const string& filename)
{
    assert_server_tasklist_map_validity(state.server_tasklist_map);
    assert_unallocated_tasks_validity(this->unallocated_tasks);
    TaskList unallocated_tasks;
    if (state.auction_task.get_id() != -1) {
        unallocated_tasks.push_back(state.auction_task);
    }
    unallocated_tasks.insert(unallocated_tasks.end(), this->unallocated_tasks.begin(), this->unallocated_tasks.end());
    json env_setting_json;
    env_setting_json["env name"] = env_name;
    env_setting_json["time step"] = state.time_step;
    env_setting_json["total time steps"] = total_time_steps;
    vector<json> server_tasklist_map_json;
    for (const auto& server_tasklist : state.server_tasklist_map) {
        const Server& server = server_tasklist.first;
        const TaskList& tasklist = server_tasklist.second;
        json server_json;
        server_json["name"] = server.get_name();
        server_json["storage capacity"] = server.get_storage_cap();
        server_json["computational capacity"] = server.get_computational_cap();
        server_json["bandwidth capacity"] = server.get_bandwidth_cap();
        vector<json> tasklist_json;
        for (const auto& task : tasklist) {
            json task_json;
            task_json["name"] = task.get_name();
            task_json["required storage"] = task.get_required_storage();
            task_json["required computational"] = task.get_required_computation();
            task_json["required results data"] = task.get_required_result_data();
            task_json["auction time"] = task.get_auction_time();
            task_json["deadline"] = task.get_deadline();
            task_json["loading progress"] = task.get_loading_progress();
            task_json["compute progress"] = task.get_computing_progress();
            task_json["sending progress"] = task.get_sending_progress();
            task_json["stage"] = task.get_stage();
            task_json["price"] = task.get_price();
            tasklist_json.push_back(task_json);
        }
        server_json["tasks"] = tasklist_json;
        server_tasklist_map_json.push_back(server_json);
    }
    env_setting_json["servers"] = server_tasklist_map_json;
    vector<json> unallocated_tasks_json;
    for (const auto& task : unallocated_tasks) {
        json task_json;
        task_json["name"] = task.get_name();
        task_json["required storage"] = task.get_required_storage();
        task_json["required computational"] = task.get_required_computation();
        task_json["required results data"] = task.get_required_result_data();
        task_json["auction time"] = task.get_auction_time();
        task_json["deadline"] = task.get_deadline();

        unallocated_tasks_json.push_back(task_json);
    }
    env_setting_json["unallocated tasks"] = unallocated_tasks_json;
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Не удалось открыть файл: " << filename << "\n";
        assert(false);
    }
    file << env_setting_json.dump();
    file.close();
}
tuple<ResourceAllocationEnvironment, EnvState> ResourceAllocationEnvironment::load_env(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Не удалось открыть файл: " << filename << "\n";
        assert(false);
    }
    json env_setting_json;
    file >> env_setting_json;
    file.close();
    string name = env_setting_json["env name"].get<string>();
    int time_step = env_setting_json["time step"].get<int>();
    int total_time_steps = env_setting_json["total time steps"].get<int>();
    ServerTaskListMap server_tasklist_map;
    int server_id = 0;
    string server_name;
    float storage_cap, computational_cap, bandwidth_cap;
    int task_id = 0, auction_time, deadline;
    string task_name;
    float required_storage, required_computation, required_result_data,
        loading_progress, computing_progress, sending_progress, price;
    TaskStage stage;
    for (const auto& server_json : env_setting_json["servers"]) {
        server_id++;
        server_name = server_json["name"].get<string>();
        storage_cap = server_json["storage capacity"].get<float>();
        computational_cap = server_json["computational capacity"].get<float>();
        bandwidth_cap = server_json["bandwidth capacity"].get<float>();
        TaskList tasklist;
        for (const auto& task_json : server_json["tasks"]) {
            task_id++;
            task_name = task_json["name"].get<string>();
            auction_time = task_json["auction time"].get<int>();
            deadline = task_json["deadline"].get<int>();
            required_storage = task_json["required storage"].get<float>();
            required_computation = task_json["required computational"].get<float>();
            required_result_data = task_json["required results data"].get<float>();
            stage = static_cast<TaskStage>(task_json["stage"].get<TaskStage>());
            loading_progress = task_json["loading progress"].get<float>();
            computing_progress = task_json["compute progress"].get<float>();
            sending_progress = task_json["sending progress"].get<float>();
            price = task_json["price"].get<float>();
            tasklist.push_back(Task(task_id, task_name, auction_time, deadline, required_storage, required_computation,
                required_result_data, loading_progress, computing_progress, sending_progress, stage, price));
        }
        Server server(server_id, server_name, storage_cap, computational_cap, bandwidth_cap);
        server_tasklist_map[server] = tasklist;
    }
    TaskList unallocated_tasks;
    for (const auto& task_json : env_setting_json["unallocated tasks"]) {
        task_id++;
        task_name = task_json["name"].get<string>();
        auction_time = task_json["auction time"].get<int>();
        deadline = task_json["deadline"].get<int>();
        required_storage = task_json["required storage"].get<float>();
        required_computation = task_json["required computational"].get<float>();
        required_result_data = task_json["required results data"].get<float>();
        unallocated_tasks.push_back(Task(task_id, task_name, auction_time, deadline,
            required_storage, required_computation, required_result_data));
    }
    ResourceAllocationEnvironment env(name, server_tasklist_map, unallocated_tasks, total_time_steps, time_step);
    return make_tuple(env, env.state);
}
tuple<string, ServerList, TaskList, int> ResourceAllocationEnvironment::load_settings(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Не удалось открыть файл: " << filename << "\n";
        assert(false);
    }
    json env_setting_json;
    file >> env_setting_json;
    file.close();
    string env_name = env_setting_json["name"].get<string>();
    if (env_name == "") {
        cout << "Имя среды не задано!\n";
        assert(false);
    }
    int server_id = 0;
    string server_name;
    float storage_cap, computational_cap, bandwidth_cap;
    int task_id = 0, auction_time, deadline;
    string task_name;
    float required_storage, required_computation, required_result_data;
    int total_time_steps = (rand() % (env_setting_json["max total time steps"].get<int>() - env_setting_json["min total time steps"].get<int>() + 1)) + env_setting_json["min total time steps"].get<int>();
    assert_total_time_steps_validity(total_time_steps);
    int max_server_num = (rand() % (env_setting_json["max total servers"].get<int>() - env_setting_json["min total servers"].get<int>() + 1)) + env_setting_json["min total servers"].get<int>() - 1;
    ServerList serverlist;
    for (; server_id <= max_server_num; ++server_id) {
        json server_json = env_setting_json["server settings"][rand() % env_setting_json["server settings"].size()];
        server_name = server_json["name"].get<string>() + " " + to_string(server_id);
        storage_cap = (rand() % (server_json["max storage capacity"].get<int>() - server_json["min storage capacity"].get<int>() + 1)) + server_json["min storage capacity"].get<int>();
        computational_cap = (rand() % (server_json["max computational capacity"].get<int>() - server_json["min computational capacity"].get<int>() + 1)) + server_json["min computational capacity"].get<int>();
        bandwidth_cap = (rand() % (server_json["max bandwidth capacity"].get<int>() - server_json["min bandwidth capacity"].get<int>() + 1)) + server_json["min bandwidth capacity"].get<int>();
        Server server(server_id, server_name, storage_cap, computational_cap, bandwidth_cap);
        server.assert_validity();
        serverlist.push_back(server);
    }
    int max_task_num = (rand() % (env_setting_json["max total tasks"].get<int>() - env_setting_json["min total tasks"].get<int>() + 1)) + env_setting_json["min total tasks"].get<int>() - 1;
    TaskList tasklist;
    for (; task_id <= max_task_num; ++task_id) {
        json task_json = env_setting_json["task settings"][rand() % env_setting_json["task settings"].size()];
        task_name = task_json["name"].get<string>() + " " + to_string(task_id);
        auction_time = rand() % (total_time_steps + 1);
        deadline = auction_time + (rand() % (task_json["max deadline"].get<int>() - task_json["min deadline"].get<int>() + 1)) + task_json["min deadline"].get<int>();
        required_storage = (rand() % (task_json["max required storage"].get<int>() - task_json["min required storage"].get<int>() + 1)) + task_json["min required storage"].get<int>();
        required_computation = (rand() % (task_json["max required computation"].get<int>() - task_json["min required computation"].get<int>() + 1)) + task_json["min required computation"].get<int>();
        required_result_data = (rand() % (task_json["max required results data"].get<int>() - task_json["min required results data"].get<int>() + 1)) + task_json["min required results data"].get<int>();
        Task task(task_id, task_name, auction_time, deadline, required_storage, required_computation, required_result_data);
        task.assert_validity();
        tasklist.push_back(task);
    }
    return make_tuple(env_name, serverlist, tasklist, total_time_steps);
}
EnvState ResourceAllocationEnvironment::reset() {
    if (this->env_settings.size() == 0) {
        cout << "Для сброса среды список настроек среды должен быть задан!\n";
        assert(false);
    }
    vector<string> env_settings = this->env_settings;
    string env_setting_filename = env_settings[rand() % env_settings.size()];
    auto load_env_setting = load_settings(env_setting_filename);
    ServerList new_servers = get<1>(load_env_setting);
    TaskList new_tasks = get<2>(load_env_setting);
    string env_name = get<0>(load_env_setting);
    int total_time_steps = get<3>(load_env_setting);
    sort ( new_tasks.begin(), new_tasks.end(),
        [](const Task& task1, const Task& task2) {
            return task1.get_auction_time() < task2.get_auction_time();
        }
    );
    TaskList unallocated_tasks(new_tasks.begin(), new_tasks.end());
    ServerTaskListMap server_tasklist_map;
    for (const auto& server : new_servers)
        server_tasklist_map[server] = TaskList();
    ResourceAllocationEnvironment env(env_name, server_tasklist_map, unallocated_tasks, total_time_steps, 0);
    *this = env;
    this->env_settings = env_settings;
    return state;
}
tuple<EnvState, ServerRewardMap, bool, unordered_map<string, string>> ResourceAllocationEnvironment::step(const ServerActionMap& actions) {
    unordered_map<string, string> info;
    for (const auto& server_tasklist : state.server_tasklist_map) {
        if (actions.find(server_tasklist.first) == actions.end()) {
            cout << "В словаре с серверами и действиями не найден сервер, который присутствует в среде!\n";
            assert(false);
        }
    }
    EnvState next_state;
    ServerRewardMap server_reward_map;
    if (state.auction_task.get_id() != -1) {
        info["step type"] = "auction";
        float min_price = numeric_limits<float>::infinity();
        float second_min_price = numeric_limits<float>::infinity();
        ServerList min_servers;
        for (const auto& server_action : actions) {
            const Server& server = server_action.first;
            const float& price = server_action.second.action.get();
            if (price > 0) {
                if (price < min_price) {
                    second_min_price = min_price;
                    min_price = price;
                    min_servers = { server };
                }
                else if (price == min_price) {
                    min_servers.push_back(server);
                    second_min_price = price;
                }
                else if (price < second_min_price) {
                    second_min_price = price;
                }
            }
        }
        next_state = EnvState(state.server_tasklist_map, next_auction_task(state.time_step), state.time_step);
        if (!min_servers.empty()) {
            Server winning_server = min_servers[rand() % min_servers.size()];
            info["min price servers"] = "[";
            for (int i = 0; i < min_servers.size(); ++i) {
                info["min price servers"] += min_servers[i].get_name();
                if (i < min_servers.size() - 1)
                    info["min price servers"] += ", ";
            }
            info["min price servers"] += "]";
            info["min price"] = to_string(min_price);
            info["second min price"] = to_string(second_min_price);
            info["winning server"] = winning_server.get_name();
            float price = (second_min_price < numeric_limits<float>::infinity()) ? second_min_price : min_price;
            server_reward_map[winning_server] = EnvReward(price);
            state.auction_task.assign_server(price, state.time_step);
            next_state.server_tasklist_map[winning_server].push_back(state.auction_task);
        }
        else {
            info["min price servers"] = "failed, no server won";
            cout << "Аукцион, неудача! "; // "Неудача. Ни один сервер не выиграл!\n";
        }
    }
    else {
        info["step type"] = "resource allocation";
        ServerTaskListMap next_server_tasklist_map;
        for (const auto& server_action : actions) {
            Server server = server_action.first;
            TaskWeightMap task_weight_map = server_action.second.task_action_map;
            if (!state.server_tasklist_map[server].empty()) {
                auto tasklist_tuple = server.allocate_resources(task_weight_map, state.time_step);
                next_server_tasklist_map[server] = get<0>(tasklist_tuple);
                server_reward_map[server] = EnvReward(get<1>(tasklist_tuple));
            }
            else {
                next_server_tasklist_map[server] = {};
                server_reward_map[server] = EnvReward({});
            }
        }
        int tasklists_size_sum = 0;
        for (const auto& server_tasklist : state.server_tasklist_map) {
            tasklists_size_sum += server_tasklist.second.size();
        }
        int next_tasklists_size_sum = 0;
        for (const auto& server_tasklist : next_server_tasklist_map) {
            next_tasklists_size_sum += server_tasklist.second.size();
        }
        int reward_tasklists_size_sum = 0;
        for (const auto& server_reward : server_reward_map) {
            reward_tasklists_size_sum += server_reward.second.size();
        }
        if (tasklists_size_sum != next_tasklists_size_sum + reward_tasklists_size_sum) {
            cout << "Кол-во задач в текущем состоянии не совпадает с кол-вом завершенных и незавершенных задач, полученных после распределения ресурсов!\n";
            assert(false);
        }
        next_state = EnvState(next_server_tasklist_map, next_auction_task(state.time_step + 1), state.time_step + 1);
        for (const auto& server_tasklist : next_state.server_tasklist_map) {
            for (const auto& task : server_tasklist.second) {
                if (task.get_auction_time() > next_state.time_step || next_state.time_step > task.get_deadline()) {
                    cout << "Текущий временной шаг активной задачи меньше времени проведения аукциона или больше крайнего срока завершения задачи!\n";
                    assert(false);
                }
            }
        }
        for (const auto& server_tasklist : next_state.server_tasklist_map) {
            for (int i = 0; i < server_tasklist.second.size(); i++) {
                if (
                    server_tasklist.second[i].get_stage() != TaskStage::LOADING &&
                    server_tasklist.second[i].get_stage() != TaskStage::COMPUTING &&
                    server_tasklist.second[i].get_stage() != TaskStage::SENDING
                ) {
                    cout << "Задачи в словаре серверов с выполняющимися задачами должны находиться на стадии загрузки, вычислений либо отправки!\n";
                    assert(false);
                }
            }
        }
    }
    for (const auto& server_tasklist : next_state.server_tasklist_map) {
        for (const auto& task : server_tasklist.second) {
            task.assert_validity();
        }
    }
    state = next_state;
    return make_tuple(state, server_reward_map, total_time_steps < state.time_step, info);
}
string ResourceAllocationEnvironment::env_to_string() const {
    return (
        "Среда распределения ресурсов в облачных вычислениях " + env_name +
        ": \nОбщее кол-во временных шагов: " + to_string(total_time_steps) +
        "\nНераспределенные задачи: " + Task::tasklist_to_string(unallocated_tasks) +
        "\n" + state.env_state_to_string() + "\n"
    );
}