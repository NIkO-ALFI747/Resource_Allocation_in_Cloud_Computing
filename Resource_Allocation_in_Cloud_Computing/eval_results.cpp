#pragma once
#include <cassert>
#include "eval_results.h"

void EvalResults::auction(const ServerActionMap& actions, const ServerRewardMap& rewards)
{
    for (const auto& server_reward : rewards) {
        const Server& server = server_reward.first;
        float price = server_reward.second.reward;
        total_winning_prices += price;
        winning_prices.push_back(price);
    }
    for (const auto& server_action : actions) {
        const Server& server = server_action.first;
        float action = server_action.second.action.get();
        auction_actions.push_back(action);
    }
    num_auctions++;
    env_attempted_tasks.back() += 1;
}
void EvalResults::resource_allocation(const ServerActionMap& actions, const ServerRewardMap& rewards)
{
    for (const auto& server_reward : rewards) {
        for (const auto& task : server_reward.second.tasklist) {
            if (task.get_stage() == TaskStage::COMPLETED) {
                num_completed_tasks++;
                env_completed_tasks.back()++;
                total_prices += task.get_price();
            }
            else if (task.get_stage() == TaskStage::FAILED) {
                num_failed_tasks++;
                env_failed_tasks.back()++;
                total_prices -= task.get_price();
            }
            else {
                cout << "� ������ �������������� ������ ���������� ������ � ����������� ����������!\n";
                assert(false);
            }
        }
    }
    for (const auto& server_action : actions) {
        const TaskWeightMap& task_action_map = server_action.second.task_action_map;
        for (const auto& task_action : task_action_map) {
            float action = task_action.second.get();
            weighting_actions.push_back(action);
        }
    }
    num_resource_allocations++;
}
void EvalResults::finished_env()
{
    env_attempted_tasks.push_back(0);
    env_completed_tasks.push_back(0);
    env_failed_tasks.push_back(0);
}
void EvalResults::save(int episode)
{
    cout << "������: " << episode << "\n\n"
        << "������ ����� ���������� ����: " << total_winning_prices << "\n"
        << "������ ����� ����: " << total_prices << "\n";
    /*if (episode % 50 == 0) {
        cout << "�������� ��������: \n";
        for (const auto& action : auction_actions) {
            cout << action << " ";
        }
        cout << "\n���������� ���������� ������: \n";
        for (const auto& price : winning_prices) {
            cout << price << " ";
        }
    }*/
    cout << "\n���-�� ������� ����������� �����: " << num_completed_tasks << "\n"
        << "���-�� ��������� �����: " << num_failed_tasks << "\n";
    float percent = static_cast<float>(num_completed_tasks + num_failed_tasks) / num_auctions;
    cout << "������� ���������� ���� �����: " << percent << "\n";
    float ratio = static_cast<float>(num_completed_tasks) / (num_failed_tasks + 1);
    cout << "C���������� ���-�� ������� ����������� ����� � ���-�� ���������: " << ratio << "\n";
    /*cout << "������� ������������: \n";
    for (const auto& action : weighting_actions) {
        cout << action << " ";
    }*/
    cout << "\n\n";
}
