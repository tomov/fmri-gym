FROM ubuntu:26.04 AS runtime-base

RUN apt-get update \
  && apt-get -y install --no-install-recommends \
       gosu \
       python3-minimal \
       python3-pip \
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
        python3-dev python3-pip python3-setuptools \
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
        autoconf libtool python3-numpy flex bison libbz2-dev \
    && apt-get autoremove \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
    && apt-get -y install \
        python3-virtualenv \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY fmri_gym ./ /src
RUN virtualenv /venv \
    && . /venv/bin/activate \
    && cd /src \
## nle/nethack compile crash in modern gcc
    && CC=/usr/bin/gcc-12 pip install --no-cache-dir -e .[nethack]  \
    && pip install --no-cache-dir -e .[dbp] \
## some of the deps above overwrite pygame-ce, not solved by pyproject.toml need to be reinstalled.
    && pip uninstall -y pygame \
    && pip install --no-cache-dir --force pygame-ce


FROM runtime-base AS runtime

COPY --from=builder /venv /venv
COPY --from=builder /src /src

ENV PATH=/venv/bin/:$PATH

WORKDIR /src

RUN playwright install chrome
