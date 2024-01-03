import torch
from triton.testing import do_bench


def get_flops(N, get_kernels=False):
    A = torch.randn(N, N, device='cuda', dtype=torch.float16)
    B = torch.randn(N, N, device='cuda', dtype=torch.float16)

    def f():
        return torch.mm(A, B)

    if get_kernels:
        with torch.profiler.profile() as prof:
            f()

        for e in prof.events():
            if "gemm" in e.name or "triton" in e.name or "gemv" in e.name:
                print(f"{N}: {e.name}")
                timer = e.cuda_time / 1e3
    # TODO: i refactor it from do_bench(f), didn't sure that it's correct.
    timer = do_bench(f, percentiles=[1.0])
    iters_per_second = 1e3 / timer[0]
    flops = A.shape[0] * A.shape[1] * B.shape[1] * 2
    flops_achieved = iters_per_second * flops / 1e12
    print(f"{N}: {flops_achieved:.2f}TF/s")


for N in range(1, 4096):
    get_flops(N)