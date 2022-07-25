#!/bin/tcsh

set script_pwd=`readlink -f "$0"`
set scriptdir=`dirname $script_pwd`

foreach II ( `seq 0 10` )
    echo $II
    foreach inp ( `ls "$scriptdir"/ptasim_input_files/spinnoise_100r/[0-9]*.inp` );
	mkdir -p data/100r/${II}/
	cd data/100r/${II}
	echo $inp
	ptaSimulate "$inp"
	set ind=`basename "$inp" .inp`
	echo $ind
	source regsamp_${ind}/scripts/runScripts_master
	cd $scriptdir
    end
end
