#pragma once
#include "env_state.h"
#include <string>

/// EnvState

EnvState::EnvState(
    const ServerTaskListMap& server_tasklist_map,
    const Task auction_task,
    const int time_step
) :
    server_tasklist_map(server_tasklist_map),
    auction_task(auction_task),
    time_step(time_step)
{}
string EnvState::env_state_to_string() const {
    string auction_task_str = (auction_task.get_id() != -1) ? auction_task.task_to_string() : "Не назначена.\n";
    string server_tasklist_map_str = server_tasklist_map.size() > 0 ? Server::server_tasklist_map_to_string(server_tasklist_map) : "Не назначено.\n";
    return (
        "Состояние среды: \nТекущий временной шаг: " + to_string(time_step) + "\n" +
        "Задача, выставленная на аукцион: \n" + auction_task_str +
        "Сервера и списки задач, выполняющихся на этих серверах: \n" + server_tasklist_map_str + "\n"
    );
}

/// EnvReward

EnvReward::EnvReward() : reward(0.0), tasklist({}) {}
EnvReward::EnvReward(float reward) : reward(reward) {}
EnvReward::EnvReward(const TaskList& tasklist) : reward(0.0), tasklist(tasklist) {}
size_t EnvReward::size() const {
    return tasklist.size();
}

/// EnvAction

EnvAction::EnvAction() : action(Price()), task_action_map({}) {}
EnvAction::EnvAction(float action) : action(action) {}
EnvAction::EnvAction(const TaskWeightMap& task_action_map) : action(Price()), task_action_map(task_action_map) {}
size_t EnvAction::size() const {
    return task_action_map.size();
}