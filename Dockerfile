FROM ubuntu:26.04 AS runtime-base

# Python is uv's standalone build, not the distribution's: the version is the project's
# (.python-version), whatever the base image ships. It lives outside /root so the venv's
# interpreter link resolves in the runtime stage too.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy
COPY .python-version /tmp/.python-version
RUN uv python install "$(cat /tmp/.python-version)" && rm /tmp/.python-version

RUN apt-get update \
  && apt-get -y install --no-install-recommends \
       gosu ca-certificates \
       libasound2t64 pulseaudio libpulse0 libportaudio2 \
       libsdl3-0 \
       x11-xserver-utils \
       libavformat62 libavcodec62 libavdevice62 libavutil60 libswscale9 libswresample6 ffmpeg x264 x265 \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -Rf /usr/share/doc

# ── Builder stage – compiles from source + installs Python deps ────────────
FROM runtime-base AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Build and development dependencies (not needed at runtime)
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        wget curl unzip \
        build-essential cmake make nasm meson ninja-build pkg-config swig \
        libusb-1.0-0-dev portaudio19-dev libasound2-dev libpulse-dev \
        qtbase5-dev \
        libglew-dev libtbb-dev \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev \
        libopencv-dev libeigen3-dev \
        libxml2-dev libglib2.0-dev gobject-introspection \
        libgtk-3-dev gtk-doc-tools xsltproc \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgirepository-2.0-dev gir1.2-girepository-2.0-dev gettext \
        libturbojpeg0-dev libvlc-dev \
        libsdl2-dev \
        git \
## require for nle compilation, crashes on more recent gcc
        gcc-12 \
        autoconf libtool flex bison libbz2-dev \
    && apt-get autoremove \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

COPY humans ./ /src
# external/ (the vgdl / coom / baba_auto submodules) comes in with the build context. An
# uninitialized submodule copies as an empty dir, so check for the files the envs read
# before the long pip step rather than failing there (baba_auto) or at run time (coom, vgdl).
RUN test -d /src/external/vgdl/games -a -d /src/external/coom/COOM/env/scenarios \
         -a -f /src/external/baba_auto/CMakeLists.txt \
    || { echo "external/ submodules are not checked out: run 'git submodule update --init' before 'docker build'" >&2; exit 1; }
## uv pip reads [tool.uv.sources], so the gym/ wrappers come from their paths in one resolve.
## nle/nethack compile crash in modern gcc; baba_auto compiles pyBaba (~30 s)
RUN uv venv /venv --python "$(cat /src/.python-version)" \
    && cd /src \
    && CC=/usr/bin/gcc-12 uv pip install --python /venv --no-cache -e '.[nethack,dbp,vgdl]' \
## some of the deps above overwrite pygame-ce, not solved by pyproject.toml need to be reinstalled.
    && uv pip uninstall --python /venv pygame \
    && uv pip install --python /venv --no-cache --reinstall pygame-ce

FROM runtime-base AS runtime

COPY --from=builder /venv /venv
COPY --from=builder /src /src

ENV PATH=/venv/bin/:$PATH
# predownload binaries
RUN python -c 'import stk_gym; stk_gym.binary._download()'

WORKDIR /src

RUN playwright install chrome

# install stk runtime deps
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        libbluetooth3 libsdl2-2.0-0 libcurl4t64 libenet7 libfreetype6 libharfbuzz-bin libjpeg-turbo8 libogg0 libopenal1 libpng16-16t64 libssl3t64 libvorbis0a libmbedtls21 zlib1g \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    ARG SUPERTUXKART_DATADIR=/root/.cache/supertuxkart-gym/gym-v0.1.1/stk
