#!/usr/bin/env bash
# Checks that HEAD is the release commit for a tag and that every version site carries its version.
#
#     .github/scripts/check-release.sh v2.5.1
set -euo pipefail
tag=$1
subject=$(git log -1 --format=%s)
if [ "$subject" != "Release: $tag" ]; then
    echo "check-release: $tag must sit on 'Release: $tag', but HEAD is '$subject'"
    exit 1
fi
version=$(printf '%s\n' "$tag" | sed -nE 's/^v([0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?)$/\1/p')
if [ -z "$version" ]; then
    echo "check-release: a release tag is exactly 'vX.Y.Z' or 'vX.Y.Z-rc.N', got '$tag'"
    exit 1
fi
stale=""
[ "$(cat VERSION)" = "$version" ] || stale="$stale VERSION"
grep -qxF "version = \"$version\"" Cargo.toml || stale="$stale Cargo.toml"
grep -A1 -xF 'name = "stringtape"' Cargo.lock | grep -qxF "version = \"$version\"" || stale="$stale Cargo.lock"
if [ -n "$stale" ]; then
    echo "check-release: these files don't carry $version:$stale"
    exit 1
fi
