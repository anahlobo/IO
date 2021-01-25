from netCDF4 import Dataset
import os
from shutil import copyfile

def make_netcdf_file(par, filename, foldername, interface_data, years, mixed_layer_data):
    combo = os.path.join(foldername, filename)
    f = Dataset(combo, "w", format="NETCDF4")

    # create the dimensions
    f.createDimension('time', years) 
    f.createDimension('lat', par.nlat) 
    f.createDimension('nBasins', par.n_B) 
    f.createDimension('nClasses', par.n_b) 
    f.createDimension('nMasses', par.max_water_masses) 
    f.createDimension('outrcop_details', 4) 
    f.createDimension('buoyancy', par.n_b+1) 

    # make the variables:
    # createVariable(varname, datatype, dimensions=())
    # http://unidata.github.io/netcdf4-python/#netCDF4.Dataset.createVariable
    f.createVariable('time', interface_data.dtype, ('time',))
    f.createVariable('depths', interface_data.dtype, ('nBasins', 'nClasses', 'nMasses', 'lat', 'time'))
    f.createVariable('adiab_flux', interface_data.dtype, ('nBasins', 'nClasses', 'lat', 'time'))
    f.createVariable('flux_diab_in', interface_data.dtype, ('nBasins', 'nClasses', 'lat', 'time'))
    f.createVariable('flux_diab_out', interface_data.dtype, ('nBasins', 'nClasses', 'lat', 'time'))
    f.createVariable('outcrops', interface_data.dtype, ('nBasins', 'nClasses', 'nMasses', 'outrcop_details', 'time'))
    f.createVariable('lat', interface_data.dtype, ('lat',))
    f.createVariable('top_slope', interface_data.dtype, ('lat',))
    f.createVariable('buoyancy', interface_data.dtype, ('buoyancy',))

    f.variables['lat'][:] = par.lat
    f.variables['top_slope'][:] = par.slopes_top
    f.variables['buoyancy'][:] = par.buoyancy
    f.close()

def save_inputs_to_file(par, filename, foldername):
    combo = os.path.join(foldername, 'input_control.py')
    copyfile('input_control.py', combo)

def save_single_step_to_netcdf_file(save_counter, par, interface_data, year, track_all_fluxes_mean, mixed_layer_data, filename, foldername, phl_final, pbs_final, sfo_final, pml_final, lf_final):
    combo = os.path.join(foldername, filename)
    f = Dataset(combo, "r+")
    f.variables['time'][save_counter] = year
    f.variables['depths'][:,:,:,:,save_counter] = interface_data
    f.variables['adiab_flux'][:,:,:,save_counter] = track_all_fluxes_mean[0,:,:,:]
    f.variables['flux_diab_in'][:,:,:,save_counter] = track_all_fluxes_mean[1,:,:,:]
    f.variables['flux_diab_out'][:,:,:,save_counter] = track_all_fluxes_mean[2,:,:,:]
    f.variables['outcrops'][:,:,:,:,save_counter] = mixed_layer_data
    f.close()