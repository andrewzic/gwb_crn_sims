#!/bin/tcsh

set psrlist=$PWD"/psrs.dat"

set basedir=$PWD

#echo path is wildcard glob str for all output dirs where you want to run runcholspec
set echo_path=$1 #"all_similar_regsamp/output/real_*"

foreach d (`echo ${echo_path}`)
    #cd $d
    #$basedir/runcholspec.csh $psrlist
    #cd $basedir
    python plot_cholspec.py ${d}"/J*.spec"
end
   
    
