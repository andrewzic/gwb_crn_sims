#!/bin/bash

export script_pwd=`readlink -f "$0"`
export scriptdir=`dirname $script_pwd`

for inp in  `ls "$scriptdir"/ptasim_input_files/spinnoise_100r/[0-9]*.inp` ; do
    #mkdir -p data/100r/${II}
    cd data/100r/
    echo $inp
    export ind=`basename "$inp" .inp`
    echo $ind
    if [ ! -d regsamp_${ind} ]
    then
	ptaSimulate "$inp" &
	export p=$!
	wait $p
    fi
    csh regsamp_${ind}/scripts/runScripts_master
    wait
    export p=$!
    wait $p
    cd $scriptdir
done
