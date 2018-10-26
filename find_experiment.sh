#!/bin/bash

if [[ "$#" != 1 ]] && [[ "$#" != 2 ]] ; then
    echo "Usage: $0 <experiment-id> [<dir>]"
    exit 1
fi

exp="$1"
if [[ "$#" == 2 ]] ; then
    dir="$2"
else
    mydir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    dir="${mydir}/Outputs/"
fi

if command -v rg >/dev/null 2>&1 ; then
    rg -g 'experiment_id.txt' ${exp} ${dir}
else
    grep -r --include='experiment_id.txt' ${exp} ${dir}
fi
