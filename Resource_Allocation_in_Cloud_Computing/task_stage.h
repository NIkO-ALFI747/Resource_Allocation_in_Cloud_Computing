#pragma once
#include<iostream>
using namespace std;
enum class TaskStage {
    UNASSIGNED,
    LOADING,
    COMPUTING,
    SENDING,
    COMPLETED,
    FAILED
};
string stage_to_string(const TaskStage& task_stage);