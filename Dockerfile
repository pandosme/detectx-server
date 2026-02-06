ARG ARCH=aarch64
ARG VERSION=12.8.0
ARG UBUNTU_VERSION=24.04
ARG REPO=axisecp
ARG SDK=acap-native-sdk

#-------------------------------------------------------------------------------
# Stage 1: TensorFlow environment (cached layer)
#-------------------------------------------------------------------------------
FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION} AS tensorflow-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for installations using pip
RUN python3 -m venv /opt/venv

# Install TensorFlow for model parameter extraction (CACHED)
RUN . /opt/venv/bin/activate && pip install --no-cache-dir tensorflow

#-------------------------------------------------------------------------------
# Stage 2: Build ACAP application
#-------------------------------------------------------------------------------
FROM tensorflow-base

WORKDIR /opt/app

# Copy application source, headers, and prebuilt libraries
COPY ./app .

# Ensure local lib and include directories exist (if not already present)
RUN mkdir -p lib include

# Create library symlinks (handles broken symlinks from git on Windows)
# This ensures the linker can find -ljpeg and -lturbojpeg regardless of platform
RUN cd lib && \
    rm -f libjpeg.so libturbojpeg.so && \
    ln -sf libjpeg.so.62.4.0 libjpeg.so && \
    ln -sf libturbojpeg.so.0.3.0 libturbojpeg.so

# Extract model parameters using TensorFlow (generates model_params.h)
RUN . /opt/venv/bin/activate && python extract_model_params.py 'model/model.tflite'

# Build and package ACAP application with assets required by your app
ARG CHIP=
RUN . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'settings/settings.json' \
    -a 'model/model.tflite' \
    -a 'model/labels.txt' \
    -a 'html'
