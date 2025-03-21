#pragma once
#include "server.h"

class TaskPricingAgent {
private:
    void assert_task_input_validity(const Task& auction_task, const TaskList& allocated_tasks, int time_step) const;
protected:
    string name;
    int limit_parallel_tasks;
    virtual float get_action(const Task& auction_task, const TaskList& allocated_tasks, const Server& server, int time_step,
        bool training = false) = 0;
public:
    TaskPricingAgent(const string& name, int limit_parallel_tasks = -1);
    float bid(const Task& auction_task, const TaskList& allocated_tasks, const Server& server, int time_step, bool training = false);
};