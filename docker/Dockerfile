FROM ubuntu:20.04
LABEL maintainer="Ryota Yoneyama:ryota.yoneyama1031@gmail.com"

SHELL ["/bin/bash", "-c"]

# Install packages without prompting the user to answer any questions
ENV DEBIAN_FRONTEND noninteractive 

# Install packages
RUN apt-get update && apt-get install -y \
    locales \
    lsb-release \
    mesa-utils \
    git \
    subversion \
    vim \
    wget \
    curl \
    htop \
    libssl-dev \
    build-essential \
    dbus-x11 \
    software-properties-common \
    gdb valgrind && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install new paramiko (solves ssh issues)
RUN apt-add-repository universe
RUN apt-get update && apt-get install -y python3-pip python build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN /usr/bin/yes | pip3 install --upgrade pip
    RUN /usr/bin/yes | pip3 install --upgrade paramiko
RUN /usr/bin/yes | pip3 install --upgrade setuptools

# Locale
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Entry script - This will also run terminator
COPY  docker/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
