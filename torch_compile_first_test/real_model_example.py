import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
opt_model = torch.compile(model, backend="inductor")

print(model(torch.randn(1, 3, 64, 64)))
# print(torch._dynamo.list_backends())
