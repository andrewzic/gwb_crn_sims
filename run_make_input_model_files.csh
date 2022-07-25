#!/bin/tcsh

foreach psr_noise_f ( `ls psr_noise_vals/*.dat` );
    echo $psr_noise_f
    set cholspec_inp_dir=data/100r/regsamp_`basename $psr_noise_f .dat`/output/cholspec_inp_files/
    echo ${cholspec_inp_dir}
    if (! -d $cholspec_inp_dir) then
	mkdir $cholspec_inp_dir
    endif
    
    
    python3 make_input_model_files.py $psr_noise_f $cholspec_inp_dir
    
end
  
