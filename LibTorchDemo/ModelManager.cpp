#include "ModelManager.h"

ModelManager::ModelManager(const char* module_name, const bool gpu_support)
    : model_name_(module_name), gpu_support_(gpu_support),
      device_(gpu_support_ && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_(nullptr)
{
}

bool ModelManager::load_model()
{
    std::cout << "Loding model";
    if (gpu_support_)
        std::cout << " with gpu support";
    std::cout << ": " << model_name_ << std::endl;

    model_ = torch::jit::load(model_name_, device_);

    std::cout << "Done!" << std::endl;

    return model_ != nullptr;
}

bool ModelManager::exec_model()
{
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({ 1, 3, 224, 224 }));

    try
    {
        // Execute the model and turn its output into a tensor.
        at::Tensor output = model_->forward(inputs).toTensor(); // TODO: GPU Support Error
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }

    return true;
}
