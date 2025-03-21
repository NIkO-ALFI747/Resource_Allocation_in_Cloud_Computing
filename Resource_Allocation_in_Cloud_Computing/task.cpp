#pragma once
#include "task.h"
#include<cassert>
#include<string>
#include <tuple>
#include "task_stage.h"

/// Task

Task::Task (
    int id,
    const string& name,
    int auction_time,
    int deadline,
    float required_storage,
    float required_computation,
    float required_result_data,

    float loading_progress,
    float computing_progress,
    float sending_progress,
    TaskStage stage,
    float price
) :
    id(id),
    name(name),
    auction_time(auction_time),
    deadline(deadline),
    required_storage(required_storage),
    required_computation(required_computation),
    required_result_data(required_result_data),

    loading_progress(loading_progress),
    computing_progress(computing_progress),
    sending_progress(sending_progress),
    stage(stage),
    price(price)
{};

void Task::assign_server(float price, int time_step) {
    if (price <= 0.0f) {
        cout << "���� ������, ����������� �������� ������ ���� ����� 0!\n";
        assert(false);
    }
    if (auction_time != time_step) {
        cout << "����� �������� ������ �� ��������� � ������� ��������� �����!\n";
        assert(false);
    }
    if (stage != TaskStage::UNASSIGNED) {
        cout << "����� ����������� ������ ������� �� ��������� ������ ���� \"�� ���������\"\n";
        assert(false);
    }
    stage = TaskStage::LOADING;
    this->price = price;
}
TaskStage Task::has_failed(const TaskStage& updated_stage, int time_step) const {
    if (time_step >= deadline && updated_stage != TaskStage::COMPLETED) return TaskStage::FAILED;
    return updated_stage;
};
void Task::allocate_loading_resources(float loading_resources, int time_step) {
    if (stage != TaskStage::LOADING) {
        cout << "������ �� ��������� � ��������� ��������!\n";
        assert(false);
    }
    if (loading_progress >= required_storage) {
        cout << "�������� �������� ������ ��������� ��� ����� ��������� ����� ��������� ��� ���������� ����������!\n";
        assert(false);
    }
    loading_progress += loading_resources;
    TaskStage task_stage = (loading_progress < required_storage) ? TaskStage::LOADING : TaskStage::COMPUTING;
    stage = has_failed(task_stage, time_step);
}
void Task::allocate_computing_resources(float computing_resources, int time_step) {
    if (stage != TaskStage::COMPUTING) {
        cout << "������ �� ��������� � ��������� ����������!\n";
        assert(false);
    }
    if (computing_progress >= required_computation) {
        cout << "�������� ���������� ������ ��������� ��� ����� ���������� ������ ����������!\n";
        assert(false);
    }
    computing_progress += computing_resources;
    TaskStage task_stage = (computing_progress < required_computation) ? TaskStage::COMPUTING : TaskStage::SENDING;
    stage = has_failed(task_stage, time_step);
}
void Task::allocate_sending_resources(float sending_resources, int time_step) {
    if (stage != TaskStage::SENDING) {
        cout << "������ �� ��������� � ��������� �������� ����������� ����������!\n";
        assert(false);
    }
    if (sending_progress >= required_result_data) {
        cout << "�������� �������� ��������� ��� ����� ���������� ������ �������������� ������!\n";
        assert(false);
    }
    sending_progress += sending_resources;
    TaskStage task_stage = (sending_progress < required_result_data) ? TaskStage::SENDING : TaskStage::COMPLETED;
    stage = has_failed(task_stage, time_step);
}
void Task::assert_validity() const {
    if (name == "unnamed" || name == "") {
        cout << "������������ ������ �� ������!\n";
        assert(false);
    }
    if (required_storage <= 0.0f || required_computation <= 0.0f || required_result_data <= 0.0f) {
        cout << "��������� ����� ��������� ��� ������� ������ ��� ����� ���������� ��� ����� ��������� ��� �������������� ������ �� ������!\n";
        assert(false);
    }
    if (auction_time >= deadline) {
        cout << "��������� ��� �������� ��������� ���� ����� �������� ����� ���������� ������!\n";
        assert(false);
    }
    if (stage == TaskStage::UNASSIGNED) {
        if (loading_progress != 0.0f || computing_progress != 0.0f || sending_progress != 0.0f) {
            cout << "�������� ��������, ���������� ��� �������� �� ������� ��� �� ����������� ������!\n";
            assert(false);
        }
        return;
    }
    if (price <= 0.0f) {
        cout << "���� ������, ����������� ������� ������ ���� ������������!\n";
        assert(false);
    }
    if (stage == TaskStage::COMPLETED) {
        if (required_result_data > sending_progress) {
            cout << "��������� ����� �������������� ������: " << to_string(required_result_data)
                << " �� �����, ����� ������ ���������, ��� ��� ��������� �������� ��������: " << to_string(sending_progress) << "!\n";
            assert(false);
        }
        return;
    }
    if (stage == TaskStage::FAILED) {
        return;
    }
    if (stage == TaskStage::LOADING) {
        if (loading_progress >= required_storage) {
            cout << "�������� �������� ������ �� ����� �������� �� ������ ���� ������ ���� ����� ���������� ������ ������� ������!\n";
            assert(false);
        }
        if (computing_progress != 0.0f || sending_progress != 0.0f) {
            cout << "�������� ���������� � �������� ������ ���� ����� ���� �� ����� ��������!\n";
            assert(false);
        }
        return;
    }
    if (required_storage > loading_progress) {
        cout << "��������� ����� ��������� ��� ������� ������: " << to_string(required_storage) 
             << " ����� ����� �������� ��� ��� ��������� �������� ��������: " << to_string(loading_progress) << "!\n";
        assert(false);
    }
    if (stage == TaskStage::COMPUTING) {
        if (computing_progress >= required_computation){
            cout << "�������� ���������� ������ �� ����� ���������� �� ������ ���� ������ ���� ����� ���������� ������ ����������!\n";
            assert(false);
        }
        if (sending_progress != 0.0f) {
            cout << "�������� �������� ������ ���� ����� ���� �� ����� ����������!\n";
            assert(false);
        }
        return;
    }
    if (required_computation > computing_progress) {
        cout << "��������� ����� ����������: " << to_string(required_computation)
            << " ����� ����� ���������� ��� ��� ��������� �������� ����������: " << to_string(computing_progress) << "!\n";
        assert(false);
    }
    if (stage != TaskStage::SENDING) {
        cout << "���� ������ �� ������������� �� ������ ���������!\n";
        assert(false);
    }
    if (sending_progress >= required_result_data) {
        cout << "�������� �������� ����������� ���������� �� ����� �������� �� ������ ���� ������ ���� ����� ���������� ������ �������������� ������!\n";
        assert(false);
    }
}

string Task::task_to_string() const {
    return (
        "������ " + name + ":\n"
        + "�������������: " + to_string(id) + "\n"
        + "��������� ��� ��� ��������: " + to_string(auction_time) + "\n"
        + "������� ��������� ��� ��� ���������� ������: " + to_string(deadline) + "\n***\n"
        + "��������� ����� ��������� ��� ������� ������: " + to_string(required_storage) + "\n"
        + "��������� ����� ����������: " + to_string(required_computation) + "\n"
        + "��������� ����� ��������� ��� �������������� ������: " + to_string(required_result_data) + "\n***\n"
        + "�������� ��������: " + to_string(loading_progress) + "\n"
        + "�������� ����������: " + to_string(computing_progress) + "\n"
        + "�������� ��������: " + to_string(sending_progress) + "\n***\n"
        + stage_to_string(stage)
        + "����������� ���������� �������� ����: " + to_string(price) + "\n"
    );
}
string Task::tasklist_to_string(const TaskList& tasklist, bool detail_flag) {
    string tasklist_str = "";
    if (detail_flag) {
        for (const auto& task : tasklist) {
            tasklist_str += task.task_to_string() + "\n";
        }
    }
    else {
        for (const auto& task : tasklist) {
            tasklist_str += task.get_name() + ", ";
        }
        if (!tasklist_str.empty()) {
            tasklist_str = tasklist_str.substr(0, tasklist_str.size() - 2);
        }
    }
    return tasklist_str + "\n";
}
string Task::task_weight_map_to_string(const TaskWeightMap& task_weight_map) {
    string task_weight_map_str = "";
    for (const auto& task_weight : task_weight_map) {
        task_weight_map_str += "Key: \n" + task_weight.first.task_to_string() + "\n";
        task_weight_map_str += "Value: " + to_string(task_weight.second.get()) + "\n";
    }
    return task_weight_map_str;
}
string Task::task_resorce_usage_map_to_string(const TaskResourceUsageMap& task_res_usage_map) {
    string task_res_usage_map_str = "";
    for (const auto& task_res_usage : task_res_usage_map) {
        task_res_usage_map_str += "Key: \n" + task_res_usage.first.task_to_string() + "\n";
        task_res_usage_map_str += "Value: " + task_res_usage.second.res_usage_to_string() + "\n";
    }
    return task_res_usage_map_str;
}
bool Task::deeply_equal(const Task& task) const {
    return (
        id == task.id &&
        name == task.name &&
        auction_time == task.auction_time &&
        deadline == task.deadline &&
        required_storage == task.required_computation &&
        required_storage == task.required_storage &&
        required_result_data == task.required_result_data &&
        loading_progress == task.loading_progress &&
        computing_progress == task.computing_progress &&
        sending_progress == task.sending_progress &&
        stage == task.stage &&
        price == task.price
    );
}
bool Task::operator==(const Task& task) const {
    return id == task.id;
}
bool Task::operator!=(const Task& task) const {
    return id != task.id;
}

int Task::get_id() const {
    return id;
}
string Task::get_name() const {
    return name;
}
int Task::get_auction_time() const {
    return auction_time;
}
int Task::get_deadline() const {
    return deadline;
}
float Task::get_required_storage() const {
    return required_storage;
}
float Task::get_required_computation() const {
    return required_computation;
}
float Task::get_required_result_data() const {
    return required_result_data;
}
float Task::get_loading_progress() const {
    return loading_progress;
}
float Task::get_computing_progress() const {
    return computing_progress;
}
float Task::get_sending_progress() const {
    return sending_progress;
}
TaskStage Task::get_stage() const {
    return stage;
}
float Task::get_price() const {
    return price;
}

/// TaskHash

int TaskHash::operator()(const Task& task) const {
    return hash<int>{}(task.get_id());
}

/// ResourceUsage

void ResourceUsage::assert_validity(const ResourceUsageType& res_usage) {
    if (
        get<0>(res_usage) < 0.0f ||
        get<1>(res_usage) < 0.0f ||
        get<2>(res_usage) < 0.0f
    ) {
        cout << "����� ������������ �������� ������ ���� ���������������!\n";
        assert(false);
    }
}
ResourceUsage& ResourceUsage::operator=(const ResourceUsageType& res_usage) {
    assert_validity(res_usage);
    this->res_usage = res_usage;
    return *this;
}
ResourceUsage::ResourceUsage() : res_usage({-1.0f, -1.0f, -1.0f}) {};
ResourceUsage::ResourceUsage(const ResourceUsageType& res_usage) {
    *this = res_usage;
}
string ResourceUsage::res_usage_to_string() const {
    return (
        "(" + to_string(get<0>(res_usage)) + ", " +
        to_string(get<1>(res_usage)) + ", " +
        to_string(get<2>(res_usage)) + ")\n"
    );
}
ResourceUsageType ResourceUsage::get_res_usage() const {
    return res_usage;
}

/// Weight

void Weight::assert_validity(float weight) {
    if (weight <= 0.0f) {
        cout << "��� ��� ������ ������ ���� �������������!\n";
        assert(false);
    }
}
Weight& Weight::operator=(const float weight) {
    assert_validity(weight);
    this->weight = weight;
    return *this;
}
Weight::Weight(float weight) {
    *this = weight;
}
float Weight::get() const {
    return weight;
}

/// Price

void Price::assert_validity(float price) {
    if (price < 0.0f) {
        cout << "���� ��� ������ ������ ���� ���������������!\n";
        assert(false);
    }
}
Price& Price::operator=(const float price) {
    assert_validity(price);
    this->price = price;
    return *this;
}
Price::Price(float price) {
    *this = price;
}
float Price::get() const {
    return price;
}
