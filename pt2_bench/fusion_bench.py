import torch
import torch._inductor.config
import time

torch._inductor.config.triton.cudagraphs = False
torch.set_float32_matmul_precision('high')


def bench(f, name=None, iters=100, warmup=5, display=True, profile=False):
    for _ in range(warmup):
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    torch.cuda.synchronize()
    begin = time.time()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()
    us_per_iter = (time.time() - begin) * 1e6 / iters
    if name is None:
        res = us_per_iter
    else:
        res = f"{name}: {us_per_iter}us"
    if display:
        print(res)
    return res


def f1(a, b, c, d):
    a = a.relu()
    b = b.tanh()
    e = a * b
    f = (c + 2).cos()
    return (e + f) * d


inp = [torch.randn(2 ** 24, device='cuda') for _ in range(4)]

f = f1
nf = torch.compile(f)
bench(lambda: f(*inp), name="eager")
bench(lambda: nf(*inp), name="PT 2.0")
