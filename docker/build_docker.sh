#! /bin/bash
#
# build_docker.sh
# Copyright (C) 2021 yuengdelahoz <yuengdelahoz@Yueng>
#
# Distributed under terms of the MIT license.
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker build -f $DIR/Dockerfile -t tensorflow-gpu-audio $DIR

echo "DONE"
