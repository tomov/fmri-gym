#!/bin/bash
# Play one fmri-play run in the image, on this machine's X display and PulseAudio server.
# The arguments are fmri-play's; FMRI_GYM_IMAGE names another image than fmri-gym:latest.
# The GPU and sound devices are passed when the machine has them, a TTY when there is one (a CI
# runner has none of the three).
opts=()
for dev in /dev/dri /dev/snd; do
  [ -e "$dev" ] && opts+=(--device="$dev")
done
[ -t 0 ] && opts+=(-it)

docker run \
  "${opts[@]}" \
  --network host \
  -v $PWD/src/:/src/task_stimuli/src/ \
  -v ${XDG_RUNTIME_DIR}/pulse:${XDG_RUNTIME_DIR}/pulse \
  -v $PWD/data:/src/data \
  -v ~/data/tests/test_docker:/data \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -e DISPLAY \
  -e XAUTHORITY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${XAUTHORITY}:${XAUTHORITY} \
  -e PUPIL_PATH=/src/pupil \
  -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
  -e XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR} \
  "${FMRI_GYM_IMAGE:-fmri-gym:latest}" \
  /venv/bin/python3 /venv/bin/fmri-play "$@"
