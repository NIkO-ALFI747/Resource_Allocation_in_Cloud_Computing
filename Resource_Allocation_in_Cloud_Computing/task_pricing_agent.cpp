#pragma once
#include "task_pricing_agent.h"
#include <cassert>

void TaskPricingAgent::assert_task_input_validity(const Task& auction_task, const TaskList& allocated_tasks, int time_step) const {
    if (auction_task.get_stage() != TaskStage::UNASSIGNED) {
        cout << "���������� ������ ������ ����� ���������: \"�� ���������\"!\n";
        assert(false);
    }
    if (auction_task.get_auction_time() != time_step) {
        cout << "����� ���������� �������� ������ ��������� � ������� ��������� �����!\n";
        assert(false);
    }
    for (const auto& task : allocated_tasks) {
        if (
            task.get_stage() != TaskStage::LOADING &&
            task.get_stage() != TaskStage::COMPUTING &&
            task.get_stage() != TaskStage::SENDING
        ) {
            cout << "������ � ������ �����, ������������� �� ��������, ������ ���������� �� ������ ��������, ���������� ���� ��������!\n";
            assert(false);
        }
        if (task.get_auction_time() > time_step || time_step > task.get_deadline()) {
            cout << "������� ��������� ��� ������ ������� ���������� �������� ��� ������ �������� ����� ���������� ������!\n";
            assert(false);
        }
    }
}
TaskPricingAgent::TaskPricingAgent(const string& name, int limit_parallel_tasks) : 
    name(name), 
    limit_parallel_tasks(limit_parallel_tasks) 
{}
float TaskPricingAgent::bid(const Task& auction_task, const TaskList& allocated_tasks, const Server& server, int time_step, bool training) {
    assert_task_input_validity(auction_task, allocated_tasks, time_step);
    if (limit_parallel_tasks == -1 || allocated_tasks.size() < limit_parallel_tasks) {
        float action = get_action(auction_task, allocated_tasks, server, time_step, training);
        if (action < 0.0f) {
            cout << "������ ��� �������� ������ ���� ���������������!\n";
            assert(false);
        }
        return action;
    }
    else {
        return 0.0f;
    }
}