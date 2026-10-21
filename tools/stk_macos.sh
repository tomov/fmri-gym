#!/bin/sh
# Build SuperTuxKart's gym engine for this Mac and assemble the pack the stk backend runs.
#
# supertuxkart-gym 0.1.3 ships a game pack for Linux x86_64 only, and refuses to fetch
# one anywhere else. The engine source is the submodule external/stk-code (the commit
# the README pins: chrplr/stk-code#3, which adds macOS packs and makes the frames
# replay). This builds it the way that PR's release job does and lays it out as a
# release pack:
#
#   1. check the submodule out (it is `update = none`, so a recursive clone skips it)
#   2. fetch upstream's prebuilt macOS dependencies, checked against DEPS_SHA256
#   3. build the engine for macOS 10.15 and later
#   4. fetch the gym-v0.1.3 Linux pack, checked against the release's SHA256SUMS,
#      for its assets (the same files on every platform)
#   5. put the macOS binary in it, with the libraries it loads (bundle_macos_libs.sh)
#
# and prints the STK_ENV_BIN line to export. The build, the dependencies and the pack
# go under OUT (default ~/.cache/supertuxkart-gym/macos), outside this checkout: they
# are about 2 GB. A rerun reuses them. Needs the Xcode command line tools
# (xcode-select --install) and cmake (brew install cmake); about 20 minutes.
#
# Usage: tools/stk_macos.sh [OUT]
set -eu

DEPS_URL=https://github.com/supertuxkart/dependencies/releases/download/preview/dependencies-macosx.tar.xz
DEPS_SHA256=4d1ee5e9f84d864d5ffee1ca6f6a4fff1025a5d833bd07a2295c429f1ff15248
RELEASE=https://github.com/chrplr/stk-code/releases/download/gym-v0.1.3
LINUX_PACK=SuperTuxKart-gym-linux-x86_64

[ "$(uname -s)" = Darwin ] || { echo "this builds the macOS engine; on Linux the wheel fetches its own" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake not found: brew install cmake" >&2; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "no command line tools: xcode-select --install" >&2; exit 1; }

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SRC=$REPO/external/stk-code
OUT=${1:-$HOME/.cache/supertuxkart-gym/macos}
DEPS=$OUT/dependencies-macosx
BUILD=$OUT/build
PACK=$OUT/SuperTuxKart-gym-macos-$(uname -m)
mkdir -p "$OUT"

git -C "$REPO" submodule update --init --checkout external/stk-code

if [ ! -d "$DEPS" ]; then
    curl -sSfL -o "$OUT/deps.tar.xz" "$DEPS_URL"
    echo "$DEPS_SHA256  $OUT/deps.tar.xz" | shasum -a 256 -c -
    tar xf "$OUT/deps.tar.xz" -C "$OUT"
fi

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_FIND_ROOT_PATH="$DEPS" -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
    -DUSE_CRYPTO_OPENSSL=FALSE -DCHECK_ASSETS=off >/dev/null
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu)"

if [ ! -d "$OUT/$LINUX_PACK" ]; then
    curl -sSfL -o "$OUT/SHA256SUMS" "$RELEASE/SHA256SUMS"
    curl -sSfL -o "$OUT/$LINUX_PACK.tar.xz" "$RELEASE/$LINUX_PACK.tar.xz"
    (cd "$OUT" && grep " $LINUX_PACK.tar.xz\$" SHA256SUMS | shasum -a 256 -c -)
    tar xf "$OUT/$LINUX_PACK.tar.xz" -C "$OUT"
fi

rm -rf "$PACK"
mkdir -p "$PACK"
cp -R "$OUT/$LINUX_PACK/stk" "$OUT/$LINUX_PACK/stk-assets" "$PACK/"
cp "$BUILD/bin/supertuxkart.app/Contents/MacOS/supertuxkart" "$PACK/supertuxkart"
sh "$SRC/tools/gym/bundle_macos_libs.sh" "$DEPS/lib" "$PACK" >/dev/null

echo
echo "Built. To play stk on this Mac, export (or put in your shell profile):"
echo "  export STK_ENV_BIN=$PACK/supertuxkart"
