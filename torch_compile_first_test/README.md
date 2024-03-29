# Requirements and setup

```bash
pip install numpy==1.24.4 torch==2.0.0 torchvision==0.15.1 timm==0.9.0 transformers==4.25.1
```

Side note: numpy starting from 1.25.0 require python3.9

Torch release - https://github.com/pytorch/pytorch/releases/tag/v2.0.0
Transformers release with PT2.0 support - https://github.com/huggingface/transformers/releases/tag/v4.25.1

# Getting Started

## Toy example

Code is in [toy_example.py](./toy_example.py)

Run with `TORCH_COMPILE_DEBUG=1` to see code generated by PyTorch using Triton backend.

```bash
TORCH_COMPILE_DEBUG=1 python toy_example.py
```

After this you can see directory [torch_compile_debug](./torch_compile_debug) and generated code in `output_code.py`
file.

