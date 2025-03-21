#pragma once
#include"task_stage.h"
#include<cassert>
string stage_to_string(const TaskStage& task_stage) {
    string task_stage_str = "Состояние задачи: ";
    switch (task_stage) {
        case TaskStage::UNASSIGNED: 
            task_stage_str += "не назначена.";
            break;
        case TaskStage::LOADING: 
            task_stage_str += "в процессе загрузки.";
            break;
        case TaskStage::COMPUTING: 
            task_stage_str += "в процессе вычислений.";
            break;
        case TaskStage::SENDING: 
            task_stage_str += "в процессе отправки результатов.";
            break;
        case TaskStage::COMPLETED: 
            task_stage_str += "успешно выполнена.";
            break;
        case TaskStage::FAILED: 
            task_stage_str += "провалена.";
            break;
        default: {
            cout << "Состояние задачи не соответствует заданным состояниям!\n";
            assert(false);
        }
    }
    return task_stage_str + "\n";
};