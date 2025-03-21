#pragma once
#include "server.h"

class ResourceWeightingAgent {
private:
    void assert_task_input_validity(const TaskList& allocated_tasks, int time_step) const;
    void assert_actions_validity(const TaskList& allocated_tasks, const TaskWeightMap& actions) const;
protected:
    string name;
    virtual TaskWeightMap get_actions(const TaskList& allocated_tasks, const Server& server, 
        int time_step, bool training = false) = 0;
public:
    ResourceWeightingAgent(const string& name = "");
    TaskWeightMap weight(const TaskList& allocated_tasks, const Server& server, int time_step, bool training = false);
};