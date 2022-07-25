#!/bin/csh


set psrlist=$1
set modeldir=$2
foreach psr ( `cat ${psrlist}` )
echo $psr
#if ( $psr == "J1713+0747" ) then
set model=${modeldir}/${psr}_input.model
echo $model
tempo2 -gr cholSpectra -f $psr.par $psr.tim -dcf $model
end
