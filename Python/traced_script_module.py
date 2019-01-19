# encoding: utf-8
# using Python 3.7

# Converting to Torch Script via Tracing

import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Serializing Script Module to a File
# This will produce a model.pt file in your working directory. We have now officially left the realm of Python and are ready to cross over to the sphere of C++.
traced_script_module.save("model.pt")
