#!/bin/csh

foreach psr ( `cat psrs.dat` )                                                                                                  
echo $psr
tempo2 -residuals -f regsamp_2.80_4.00/output/real_0/$psr.par regsamp_2.80_4.00/output/real_0/$psr.tim
mv -f ./residuals.dat regsamp_2.80_4.00/output/real_0/$psr.res
cat regsamp_2.80_4.00/output/real_0/$psr.tim | awk 'FNR > 2 {print $3}' > regsamp_2.80_4.00/output/real_0/$psr.mjd
paste regsamp_2.80_4.00/output/real_0/$psr.res regsamp_2.80_4.00/output/real_0/$psr.mjd >! regsamp_2.80_4.00/output/real_0/$psr.resmjd
end
