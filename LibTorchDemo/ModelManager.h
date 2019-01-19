#pragma once

#include <iostream>
#include <memory>
#include <torch/script.h>
#include <torch/cuda.h>

class ModelManager
{
public:
    explicit ModelManager(const char* module_name, const bool gpu_support = true);
    ~ModelManager() = default;

    bool load_model();
    bool exec_model();

private:
    const char* model_name_;
    const bool gpu_support_;
    torch::Device device_;
    std::shared_ptr<torch::jit::script::Module> model_;
};
