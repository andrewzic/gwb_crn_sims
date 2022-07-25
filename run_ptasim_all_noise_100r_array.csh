#!/bin/tcsh

set script_pwd=`readlink -f "$0"`
set scriptdir=`dirname $script_pwd`

set filearr=( `ls -v "$scriptdir"/ptasim_input_files/spinnoise_100r/[0-9]*.inp` );
set ind=$argv[1]
set inpfile=${filearr[$ind]}
cd ${scriptdir}/data/100r/
echo $inpfile
ptaSimulate "$inpfile"
set pref=`basename "$inpfile" .inp`
echo $pref
source regsamp_${pref}/scripts/runScripts_master
