# syntax=docker/dockerfile:1
ARG FLOWDAPT_VERSION=local
# First stage: builder environment
FROM ghcr.io/emergentmethods/flowdapt:$FLOWDAPT_VERSION AS builder

ARG NON_ROOT_USER=flowdapt
USER root

WORKDIR /srv/flowdapt_nowcast_plugin

# Install poetry
RUN pip install poetry

# Copy the project files and install dependencies
ADD flowdapt_nowcast_plugin ./flowdapt_nowcast_plugin
COPY pyproject.toml README.md ./

# Use the virtual environment from the base image to install the plugin and dependencies
RUN . /opt/venv/bin/activate && \
    poetry config installer.max-workers 4 && \
    poetry install --without dev

# Second stage: production build
FROM ghcr.io/emergentmethods/flowdapt:$FLOWDAPT_VERSION AS production

# Set up the working directory
WORKDIR /srv/flowdapt_nowcast_plugin

# Copy the plugin code
COPY --from=builder --chown=${NON_ROOT_USER}:${NON_ROOT_USER} /srv/flowdapt_nowcast_plugin /srv/flowdapt_nowcast_plugin
# Since the virtual environment has been modified in the builder stage, copy it over
COPY --from=builder --chown=${NON_ROOT_USER}:${NON_ROOT_USER} /opt/venv /opt/venv

USER ${NON_ROOT_USER}
