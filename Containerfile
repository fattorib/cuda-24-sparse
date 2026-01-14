FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
    git \
    sudo \
    gcc-11 \
    g++-11 \
    clang-11 \
    clang-format \
    wget \
    xz-utils

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

RUN wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.8.1.1_cuda12-archive.tar.xz -O /tmp/cusparselt.tar.xz && \
    mkdir -p /usr/local/cusparselt && \
    tar -xf /tmp/cusparselt.tar.xz -C /usr/local/cusparselt --strip-components=1 && \
    rm /tmp/cusparselt.tar.xz && \
    if [ -d /usr/local/cusparselt/lib64 ] && [ ! -d /usr/local/cusparselt/lib ]; then \
        ln -s lib64 /usr/local/cusparselt/lib; \
    fi && \
    LIBDIR=$([ -d /usr/local/cusparselt/lib64 ] && echo lib64 || echo lib) && \
    echo "/usr/local/cusparselt/$LIBDIR" > /etc/ld.so.conf.d/cusparselt.conf && \
    ldconfig


ARG USER=user
ARG UID=1000
ARG GID=1000

RUN groupadd -f -g ${GID} ${USER} && \
    useradd -m -u ${UID} -g ${GID} -G sudo -s /bin/bash ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER ${USER}
ENV USER=${USER} SHELL=/bin/bash
WORKDIR /home/${USER}/sparsegemms