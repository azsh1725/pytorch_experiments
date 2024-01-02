FROM ghcr.io/pytorch/pytorch-nightly:2.0.0.dev20230301-devel

RUN apt update && apt install -y vim

WORKDIR /compile_tutorial

COPY toy_example.py toy_example.py

CMD ["/bin/bash"]