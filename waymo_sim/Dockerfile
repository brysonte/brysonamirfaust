# Explicitly request the ARM64 platform
FROM --platform=linux/arm64 ros:humble-ros-base

# Install system dependencies AND CycloneDDS
RUN apt-get update && apt-get install -y \
    python3-pip git curl qemu-user-static \
    ros-humble-foxglove-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Install JAX and Waymax
RUN pip3 install --upgrade "jax[cpu]"
RUN pip3 install git+https://github.com/waymo-research/waymax.git@main

# Set the working directory
WORKDIR /waymo_project

# Permanently lock the network to CycloneDDS and Localhost
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_LOCALHOST_ONLY=1

# Tell the container to speak ROS 2 on startup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install system dependencies AND CycloneDDS
RUN apt-get update && apt-get install -y \
    python3-pip git curl qemu-user-static nano \
    ros-humble-foxglove-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*