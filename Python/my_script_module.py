# encoding: utf-8
# using Python 3.7

# Converting to Torch Script via Annotation

import torch

class LegacyModel(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        def forward(self, input):
            if input.sum() > 0:
                output = self.weight.mv(input)
            else:
                output = self.weight + input

            return output

class ScriptModel(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(ScriptModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        @torch.jit.script_method
        def forward(self, input):
            if input.sum() > 0:
                output = self.weight.mv(input)
            else:
                output = self.weight + input

            return output

my_script_module = ScriptModel()

# Serializing Script Module to a File
# This will produce a model.pt file in your working directory. We have now officially left the realm of Python and are ready to cross over to the sphere of C++.
my_script_module.save("model.pt")
