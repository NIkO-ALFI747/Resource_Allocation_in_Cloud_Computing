#pragma once
#include"task_stage.h"
#include<cassert>
string stage_to_string(const TaskStage& task_stage) {
    string task_stage_str = "��������� ������: ";
    switch (task_stage) {
        case TaskStage::UNASSIGNED: 
            task_stage_str += "�� ���������.";
            break;
        case TaskStage::LOADING: 
            task_stage_str += "� �������� ��������.";
            break;
        case TaskStage::COMPUTING: 
            task_stage_str += "� �������� ����������.";
            break;
        case TaskStage::SENDING: 
            task_stage_str += "� �������� �������� �����������.";
            break;
        case TaskStage::COMPLETED: 
            task_stage_str += "������� ���������.";
            break;
        case TaskStage::FAILED: 
            task_stage_str += "���������.";
            break;
        default: {
            cout << "��������� ������ �� ������������� �������� ����������!\n";
            assert(false);
        }
    }
    return task_stage_str + "\n";
};