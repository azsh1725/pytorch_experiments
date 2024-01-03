import torch

torch.set_float32_matmul_precision('high')
import torch._inductor.config

torch._inductor.config.debug = True


def bench(f, name=None, iters=100, warmup=5, display=True, profile=False):
    import time
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
        res = f"{name}: {us_per_iter:.3f}us"

    if display:
        print(res)
    return res


def get_bandwidth(name, f):
    iters_per_second = 1e6 / bench(f, display=False)
    bytes_accessed = N ** 2 * 4 * 3
    print(f"{name}: {iters_per_second * bytes_accessed / 1e9:.2f}GB")


N = 2 ** 14


def f(a, b):
    return a + b


A = torch.randn(N, N, device='cuda')
B = torch.randn(N, N, device='cuda')

# eager: 1389.84GB
get_bandwidth("eager", lambda: f(A, B))
# torch.compile: 1388.19GB
get_bandwidth("torch.compile", lambda: torch.compile(f)(A, B))


def f2(a, b):
    return a + b.t()


A = torch.randn(N, N, device='cuda')
B = torch.randn(N, N, device='cuda')

# eager: 904.01GB
get_bandwidth("eager", lambda: f2(A, B))
# torch.compile: 1334.89GB (more is better)
get_bandwidth("torch.compile", lambda: torch.compile(f2)(A, B))
