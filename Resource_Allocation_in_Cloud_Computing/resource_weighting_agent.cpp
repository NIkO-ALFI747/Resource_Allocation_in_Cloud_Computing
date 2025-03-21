#pragma once
#include "resource_weighting_agent.h"
#include <cassert>
#include <string>

void ResourceWeightingAgent::assert_task_input_validity(const TaskList& allocated_tasks, int time_step) const {
    for (const auto& task : allocated_tasks) {
        if (
            task.get_stage() != TaskStage::LOADING &&
            task.get_stage() != TaskStage::COMPUTING &&
            task.get_stage() != TaskStage::SENDING
            ) {
            cout << "«адачи в списке задач, выполн€ющихс€ на серверах, должны находитьс€ на стадии загрузки, вычислений либо отправки!\n";
            assert(false);
        }
        if (task.get_auction_time() > time_step || time_step > task.get_deadline()) {
            cout << "“екущий временной шаг меньше времени проведени€ аукциона или больше крайнего срока завершени€ задачи!\n";
            assert(false);
        }
    }
}
void ResourceWeightingAgent::assert_actions_validity(const TaskList& allocated_tasks, const TaskWeightMap& actions) const {
    for (const auto& task_action : actions) {
        if (find(allocated_tasks.begin(), allocated_tasks.end(), task_action.first) == allocated_tasks.end()) {
            cout << "¬ списке выполн€ющихс€ на сервере задач не найдена задача, которой присвоили вес!\n";
            assert(false);
        }
        if (task_action.second.get() > 1.0f) {
            cout << "¬еса ресурсов дл€ задач должны находитьс€ в интервале от 0 до 1!\n";
            assert(false);
        }
    }
}
ResourceWeightingAgent::ResourceWeightingAgent(const string& name) : name(name) {}
TaskWeightMap ResourceWeightingAgent::weight(const TaskList& allocated_tasks, const Server& server, int time_step, bool training) {
    assert_task_input_validity(allocated_tasks, time_step);
    if (allocated_tasks.size() <= 1) {
        TaskWeightMap actions;
        for (const auto& task : allocated_tasks) {
            actions[task] = 1.0f;
        }
        return actions;
    }
    else {
        TaskWeightMap actions = get_actions(allocated_tasks, server, time_step, training);
        if (allocated_tasks.size() != actions.size()) {
            cout << "–азмер списка весов не совпадает с размером списка активных задач сервера!\n";
            assert(false);
        }
        float weights_sum = 0.0f;
        for (const auto& task_action : actions) {
            weights_sum += task_action.second.get();
        }
        if (weights_sum == 0.0f) {
            cout << "—умма весов ресурсов дл€ распределени€ равна нулю!\n";
            assert(false);
        }
        for (auto& task_action : actions) {
            task_action.second = task_action.second.get() / weights_sum;
        }
        assert_actions_validity(allocated_tasks, actions);
        return actions;
    }
}
