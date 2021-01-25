from IO_main import run
import numpy as np
import input_control 
from utilities import gen_run, welcome_message, make_working_folder

def main():
    """ 
    Constants are defined in PAR, and can be altered in the input file. 
    Initial conditions can also be defined in the input file, and restarts are triggered by setting inityn='True'. 
    """
    welcome_message()   

    run_name = 'run_name'
    netcdf_ending = '.nc'
    folder_ending = '_all'  
    filename = run_name + netcdf_ending
    foldername = run_name + folder_ending

    print('>o  Will save outputs in a netCDF4 file named: ',filename)
    make_working_folder(foldername)
    
    [par, interface_data, mixed_layer_data] = input_control.inputs()
    assert par.save_freq > par.dt_day, " >o  Error in Input File ~~~ save frequency must be larger than time-step (its not really a frequency...)"
    print('>o  Starting run.')     
    run(par, filename, foldername, interface_data, mixed_layer_data)    
    print('>o  Done.')


if __name__ == "__main__":
    main()
