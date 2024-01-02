class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[10000], arg1_1: f32[10000]):
        # File: toy_example.py:5, code: a = torch.sin(x).cuda()
        sin: f32[10000] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
        
        # File: toy_example.py:6, code: b = torch.sin(y).cuda()
        sin_1: f32[10000] = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
        
        # File: toy_example.py:7, code: return a + b
        add: f32[10000] = torch.ops.aten.add.Tensor(sin, sin_1);  sin = sin_1 = None
        return (add,)
        