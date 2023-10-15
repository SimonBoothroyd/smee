FROM --platform=linux/x86_64 condaforge/mambaforge:latest

RUN    apt update                               \
    && apt install -y git make build-essentials \
    && rm -rf /var/lib/apt/lists/*
