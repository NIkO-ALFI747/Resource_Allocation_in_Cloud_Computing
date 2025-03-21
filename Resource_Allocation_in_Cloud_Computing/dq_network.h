#pragma once
#include <torch/torch.h>
#include "rl_agent.h"

struct LSTMModel : torch::nn::Module {
    torch::nn::LSTM lstm;
    torch::nn::ReLU relu;
    torch::nn::Linear linear;
    LSTMModel(int input_width, int lstm_width, int relu_width, int num_actions)
        : lstm(torch::nn::LSTMOptions(input_width, lstm_width).batch_first(true)),
        relu(torch::nn::ReLU()),
        linear(relu_width, num_actions) {
        lstm->to(device_);
        relu->to(device_);
        linear->to(device_);
        register_module("lstm", lstm);
        register_module("relu", relu);
        register_module("linear", linear);
    }
    torch::Tensor forward(torch::Tensor x) {
        lstm->flatten_parameters();
        auto output_hidden = lstm->forward(x);
        auto output = std::get<0>(output_hidden);
        x = relu->forward(output);
        x = linear->forward(x);
        return x;
    }
};
struct BidirectionalLSTMModel : torch::nn::Module {
    torch::nn::LSTM forward_lstm;
    torch::nn::LSTM reverse_lstm;
    torch::nn::ReLU relu;
    torch::nn::Linear linear;
    BidirectionalLSTMModel(int input_width, int lstm_width, int relu_width, int num_actions)
        : forward_lstm(torch::nn::LSTMOptions(input_width, lstm_width).batch_first(true)),
        reverse_lstm(torch::nn::LSTMOptions(lstm_width, lstm_width).batch_first(true).bidirectional(true)),
        relu(torch::nn::ReLU()),
        linear(relu_width, num_actions) {
        forward_lstm->to(device_);
        reverse_lstm->to(device_);
        relu->to(device_);
        linear->to(device_);
        register_module("forward_lstm", forward_lstm);
        register_module("reverse_lstm", reverse_lstm);
        register_module("relu", relu);
        register_module("linear", linear);
    }
    torch::Tensor forward(torch::Tensor x) {
        int lstm_width = forward_lstm->options.hidden_size();
        forward_lstm->flatten_parameters();
        auto output_hidden = forward_lstm->forward(x);
        auto output = std::get<0>(output_hidden);
        reverse_lstm->flatten_parameters();
        output_hidden = reverse_lstm->forward(output);
        output = std::get<0>(output_hidden);
        x = output.slice(2, lstm_width, 2 * lstm_width);
        x = relu->forward(x);
        x = linear->forward(x);
        return x;
    }
};

inline torch::nn::Sequential create_lstm_dq_network(int input_width, int num_actions,
    int lstm_width = 64, int relu_width = 64) {
    LSTMModel model(input_width, lstm_width, relu_width, num_actions);
    return torch::nn::Sequential(model);
}
inline torch::nn::Sequential create_bidirectional_dq_network(int input_width, int num_actions,
    int lstm_width = 64,int relu_width = 64) {
    BidirectionalLSTMModel model(input_width, lstm_width, relu_width, num_actions);
    return torch::nn::Sequential(model);
}