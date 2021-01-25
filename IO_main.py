"""
Icy World ocean model
Initial design date: Oct. 26, 2018 

Author: Ana Lobo
Contact information available at anahelenalobo.com

Notes on notation:
nB & n_B refer to number of Basins (only tested for 1 basin)
nb & n_b refer to number of bodies (run with at least 3 water masses/bodies)
"""

import numpy as np
import numpy.matlib as npm
import math
from tqdm import tqdm
from make_netcdf_file import save_single_step_to_netcdf_file, make_netcdf_file, save_inputs_to_file
from melt_and_brine_functions import get_melt_and_brine_values
from update_intercept_scripts import update_intercepts
from melt_and_brine_functions import find_nearest_grid_point_south, find_nearest_grid_point_north


def run(par, filename, foldername, interface_data, mixed_layer_data):
    if par.save_freq < par.dt_day:
        raise StopIteration('Error: Save frequency must be smaller than time step.')
    
    # INITIAL CONDITIONS:
    interface_data, mixed_layer_data = update_intercepts(par, interface_data,  mixed_layer_data)
    save_inputs_to_file(par, filename, foldername)

    # TIME STEPPING:
    daytosecond = par.hoursinday * 60* 60
    tfin = par.tfin_day * daytosecond
    dt = par.dt_day * daytosecond
    save_freq_s = par.save_freq * daytosecond 
    saver = math.ceil(tfin / save_freq_s) 

    # FIRST TIME STEP:
    ttt = np.arange(0, tfin+1, dt) 
    tl = np.size(ttt)
    interface_saver = np.zeros([par.n_B, par.n_b, par.max_water_masses, par.nlat, saver])
    mixed_layer_saver = np.zeros([par.n_B, par.n_b, par.max_water_masses, 4, saver]) 
    track_all_fluxes_all = np.zeros([3, par.n_B, par.n_b, par.nlat])
    timey = np.zeros([saver])
    interface_saver[:,:,:,:,0] = interface_data
    mixed_layer_saver[:,:,:,:,0] = mixed_layer_data
    dz_dt_all, dyintercept_dt, phi_from_sides, w_eddy_in, w_eddy_out, phi_bottom_layer, phi_b_strm, surf_flux_out, phi_in_mixed_layer, latteral_flow = time_step(interface_data, mixed_layer_data, par)

    interface_data = add_time_step_interface(par, interface_data, dz_dt_all, dt)
    mixed_layer_data = add_time_step_mixedlayer(par, mixed_layer_data, dyintercept_dt, dt)
    interface_data, mixed_layer_data = update_intercepts(par, interface_data,  mixed_layer_data)

    assert phi_from_sides.shape[2] == 1, "Violated layer configurations."
    track_all_fluxes_all[0,:,:,:] = phi_from_sides[:,:,0,:]
    track_all_fluxes_all[1,:,:,:] = w_eddy_in[:,:,0,:] 
    track_all_fluxes_all[2,:,:,:] = w_eddy_out[:,:,0,:]    

    # INITIALIZE NETCDF FILE:
    make_netcdf_file(par, filename, foldername, interface_data, saver, mixed_layer_data)
    
    save_single_step_to_netcdf_file(0, par, interface_data, 0, track_all_fluxes_all, mixed_layer_data, filename, foldername, phi_bottom_layer, phi_b_strm, surf_flux_out, phi_in_mixed_layer, latteral_flow)
    next_save = save_freq_s 
    save_counter = 1
    for i in tqdm(np.arange(1, tl-1)):
        track_all_fluxes_all = np.zeros([3, par.n_B, par.n_b, par.nlat])
        out1_z, out1_y, phi_1, w_in_1, w_out_1, pbl1, pbs1, sfo1, pml1, lf1 = time_step(interface_data, mixed_layer_data, par)
        in2_z = add_time_step_interface(par, interface_data, out1_z/2, dt)
        in2_y = add_time_step_mixedlayer(par, mixed_layer_data, out1_y/2, dt)
        in2_z, in2_y = update_intercepts(par, in2_z, in2_y) 

        out2_z, out2_y, phi_2, w_in_2, w_out_2, pbl2, pbs2, sfo2, pml2, lf2 = time_step(in2_z, in2_y, par)
        in3_z = add_time_step_interface(par, interface_data, out2_z/2, dt)
        in3_y = add_time_step_mixedlayer(par, mixed_layer_data, out2_y/2, dt)
        in3_z, in3_y = update_intercepts(par, in3_z,  in3_y)

        out3_z, out3_y, phi_3, w_in_3, w_out_3, pbl3, pbs3, sfo3, pml3, lf3 = time_step(in3_z, in3_y, par)
        in4_z = add_time_step_interface(par, interface_data, out3_z/2, dt)
        in4_y = add_time_step_mixedlayer(par, mixed_layer_data, out3_y/2, dt)
        in4_z, in4_y = update_intercepts(par, in4_z,  in4_y)
      
        out4_z, out4_y, phi_4, w_in_4, w_out_4, pbl4, pbs4, sfo4, pml4, lf4 = time_step(in4_z, in4_y, par)
        
        dvar_dt = (1/6)*(out1_z + 2*out2_z + 2*out3_z + out4_z)
        interface_data = add_time_step_interface(par, interface_data, dvar_dt, dt)
        dvar_dt = (1/6)*(out1_y + 2*out2_y + 2*out3_y + out4_y)
        mixed_layer_data = add_time_step_mixedlayer(par, mixed_layer_data, dvar_dt, dt)

        phl_final = (1/6)*(pbl1 + 2*pbl2 + 2*pbl3 + pbl4)
        pbs_final = (1/6)*(pbs1 + 2*pbs2 + 2*pbs3 + pbs4)
        sfo_final = (1/6)*(sfo1 + 2*sfo2 + 2*sfo3 + sfo4)
        pml_final = (1/6)*(pml1 + 2*pml2 + 2*pml3 + pml4)
        lf_final = (1/6)*(lf1 + 2*lf2 + 2*lf3 + lf4)        
        phi_t = (1/6)*(phi_1 + 2*phi_2 + 2*phi_3 + phi_4)
        track_all_fluxes_all[0,:,:,:] =  phi_t[:,:,0,:]         
        w_in_t = (1/6)*(w_in_1 + 2*w_in_2 + 2*w_in_3 + w_in_4)
        track_all_fluxes_all[1,:,:,:] = w_in_t[:,:,0,:] 
        w_out_t = (1/6)*(w_out_1 + 2*w_out_2 + 2*w_out_3 + w_out_4)
        track_all_fluxes_all[2,:,:,:] = w_out_t[:,:,0,:]         

        interface_data, mixed_layer_data = update_intercepts(par, interface_data, mixed_layer_data) 
      
        if ttt[i] >= next_save:
            # this loop only saves results on a certain time interval, prescribed by 'save_freq'
            interface_saver[:,:,:,:, save_counter] = interface_data
            mixed_layer_saver[:,:,:,:,save_counter] = mixed_layer_data
            timey[save_counter] = ttt[i] / (par.hoursinday * par.daysinyear * 60 * 60) 
            next_save = next_save + save_freq_s
            save_single_step_to_netcdf_file(save_counter, par, interface_data, (ttt[i] / (par.hoursinday * par.daysinyear * 60 * 60)), track_all_fluxes_all, mixed_layer_data, filename, foldername, phl_final, pbs_final, sfo_final, pml_final, lf_final)
            save_counter = save_counter + 1
            
    return



def time_step(interfaces, mx, par):
    slopes, slopes_surf, surf_flux_out, dz_dy_outcrops = get_slopes(par, interfaces, mx)
    [phi_in_mixed_layer, buoyancy_flux_each_layer, phi_b_strm, total_fluxes_into_layers] = manage_melt_and_brine_functions(par, mx, interfaces, surf_flux_out, slopes_surf)
    
    slopes_combo = np.zeros([par.n_B, par.n_b, par.max_water_masses, 2])
    slopes_combo[:,:,:,0] = 10000*(2*np.cos(mx[:,:,:,1]*np.pi/180)*np.sin(mx[:,:,:,1]*np.pi/180))/par.radius 
    slopes_combo[:,:,:,1] = 10000*(2*np.cos(mx[:,:,:,3]*np.pi/180)*np.sin(mx[:,:,:,3]*np.pi/180))/par.radius 
    # slopes_combo is specific to the outcrops. 

    dyintercept_dt_meters = (1/par.d_mixed_layer)*(-par.tau/(par.rho0*par.f) + par.K*(slopes_surf-slopes_combo) - phi_b_strm)
    dyintercept_dt = 360*dyintercept_dt_meters / (2*np.pi*par.radius)    
    dz_dt_all = np.zeros([par.n_B, par.n_b, par.max_water_masses, par.nlat])

    assert par.n_B == 1, "Error: Not ready for multi-basin yet."
    dz_dt_all, phi_from_sides, w_eddy_in, w_eddy_out, phi_bottom_layer, latteral_flow = get_fluxes(par, 0, interfaces, mx, total_fluxes_into_layers, slopes, dz_dy_outcrops) 

    return dz_dt_all, dyintercept_dt, phi_from_sides, w_eddy_in, w_eddy_out, phi_bottom_layer, buoyancy_flux_each_layer, total_fluxes_into_layers, phi_in_mixed_layer, latteral_flow



def manage_melt_and_brine_functions(par, mx, interfaces, surf_flux_out, slopes_surf):
    """ Function calcultes melt flux and brine flux at the surface. """

    if par.surfacefluxes == 'primary':
        [phi_buoyancy, buoyancy_fluxes_into_layers, phi_b_strm, total_fluxes_into_layers] = get_melt_and_brine_values(par, mx, interfaces, surf_flux_out, slopes_surf)
    else: 
        print('Error: You need to define the melt and brine function being used')
        exit(1)
    
    return phi_buoyancy, buoyancy_fluxes_into_layers, phi_b_strm, total_fluxes_into_layers


def merge_slopes(par, slopes):
    output = np.zeros([par.n_B, par.n_b, par.nlat-1])
    for nB in range(par.n_B):
        for nb in range(par.n_b):
            for nlat in range(par.nlat-1):
                each_layer = slopes[nB, nb, :, nlat]
                if np.all(each_layer == par.slopes_top[nlat]):
                    take = each_layer[0]
                elif par.lat[nlat] == 0 and np.all(each_layer == 0):
                    take = 0
                elif par.lat[nlat] == 0:
                    clear_eq = np.where(each_layer == 0, np.nan, each_layer)
                    take = np.nanmin(clear_eq)
                elif np.sum(np.isnan(each_layer)) == each_layer.shape[0]:
                    take = np.nan
                elif np.sum(np.isnan(each_layer)) == each_layer.shape[0] - 1: 
                    take = np.nanmin(each_layer)   
                else:
                    clear1 = np.where(each_layer == par.slopes_top[nlat], np.nan, each_layer)
                    clear_each_layer = np.where(clear1 == par.slopes_top[nlat], np.nan, clear1)
                    if np.sum(np.isnan(clear_each_layer)) == each_layer.shape[0] - 1: 
                        take = np.nanmin(clear_each_layer)
                    else: take = np.mean(clear_each_layer)
                output[nB, nb, nlat] = take
    return output

def merge_depths(par, depths):
    """ output = merge_depths(par, depths) """
    output = np.zeros([par.n_B, par.n_b, par.nlat])
    for nB in range(par.n_B):
        for nb in range(par.n_b):
            for nlat in range(par.nlat):
                each_layer = depths[nB, nb, :, nlat]
                if np.sum(np.isnan(each_layer)) == each_layer.shape[0]:
                    take = np.nan            
                elif np.sum(np.isnan(each_layer)) == 1: 
                    take = np.nanmin(each_layer)
                elif each_layer.shape[0] == 1 and np.sum(np.isnan(each_layer)) == 0:
                    take = np.nanmin(each_layer)
                else: raise StopIteration("Something went wrong merging depths.")
                output[nB, nb, nlat] = take
    return output


def get_fluxes(par, nB, interfaces, mx, phi_surf_in_out, slopes, dz_dy_outcrops):    
    # adiabatic flux: 
    assert par.n_B == 1, "If you want multibasin, need to edit here."
    nB = 0

    phi_vector1 = np.zeros([(par.n_b+1), par.nlat-1])
    phi_vector2 = np.zeros([(par.n_b+1), par.nlat-1])

    if interfaces.shape[2] != 1:
        slopes_merged_t = merge_slopes(par, slopes)
        slopes_merged_t_ios = merge_slopes(par, dz_dy_outcrops)
        depths_merged_t = merge_depths(par, interfaces)
        slopes_merged = slopes_merged_t[nB, :,:]
        depths_merged = depths_merged_t[nB, :,:]
        slopes_merged_ios = slopes_merged_t_ios[nB, :,:]
    else:
        slopes_merged = slopes[0,:,0,:]
        slopes_merged_ios = dz_dy_outcrops[0,:,0,:]
        depths_merged = interfaces[0,:,0,:]
    
    top_vector = par.slopes_mid_top
    bottom_vector = (par.slopes_bottom[:-1] + par.slopes_bottom[1:])/2
    phi_vector1[0,:] = top_vector 
    phi_vector2[-1,:] = bottom_vector 
    phi_vector1[1:, :] = slopes_merged_ios[:,:]
    phi_vector2[:-1, :] = slopes_merged[:,:]
    # For now, the bottom has a flat slope.
    
    phi_from_sides = -par.K * (phi_vector1 - phi_vector2)
    width_mid = abs(2*np.pi*par.radius*np.cos(par.midlat * np.pi/180)) 
    width_t_mid = npm.repmat(width_mid, 1, (par.n_b+1)*(par.max_water_masses)) 
    width_mid_final = width_t_mid.reshape(par.n_b+1, par.midlat.shape[0]) 

    phi_from_sides_south = np.zeros([par.n_b+1, par.nlat]) 
    phi_from_sides_south[:,1:] = phi_from_sides * width_mid_final 
    phi_from_sides_north = np.zeros([par.n_b+1, par.nlat]) 
    phi_from_sides_north[:,:-1] = phi_from_sides * width_mid_final 

    width = abs(2*np.pi*par.radius*np.cos(par.rlat))
    phi_all =  (phi_from_sides_south[:,:] - phi_from_sides_north[:,:]).reshape((par.n_B, (par.n_b+1),par.max_water_masses,par.nlat))
    phi_bottom_layer = phi_all[:,:,:,:] 
    phi_bottom_layer[:,:-1,:,:] = (phi_bottom_layer[:,:-1,:,:] * (1 - np.isnan(interfaces))) + phi_surf_in_out 
    net_phi_from_sides = phi_bottom_layer[:,:-1,:,:] 

    # Diffusive fluxes:
    z_vector1 = np.zeros([(par.n_b+1), par.nlat])
    z_vector2 = np.zeros([(par.n_b+1), par.nlat])
    z_vector1[0,:] = par.ice_depth - par.d_mixed_layer 
    z_vector2[-1,:] = z_vector2[-1,:] + par.bottom
    z_vector1[1:,:] = depths_merged[:,:]
    z_vector2[:-1,:] = depths_merged[:,:]
    z_vector1, z_vector2 = get_delz(par, z_vector1, z_vector2)
    delz = z_vector1 - z_vector2
    
    Keddy_v1 = par.kappa - 0.5*par.delkappa*(np.tanh((interfaces - par.difftrans)*(1/par.diffthick)) + 1)

    # Adding in plumes:
    lat_2d_t  = npm.repmat(par.lat, 1, (par.n_b)*(par.max_water_masses))
    lat_2d_t2 = lat_2d_t.reshape((par.n_B, (par.n_b), (par.max_water_masses), par.nlat))
    melt_s = par.y_melt_top[0,0,0,0,0]
    melt_n = par.y_melt_top[0,0,0,0,1]
    ice_2d_t  = npm.repmat(par.ice_depth, 1, (par.n_b)*(par.max_water_masses))
    ice_2d_t2 = ice_2d_t.reshape((par.n_B, (par.n_b), (par.max_water_masses), par.nlat))
    termz_t = par.kappa*(1 - 1*np.exp(-1*((interfaces - ice_2d_t2 - par.bottom )*(1/10000))**2)) 
    termz = np.ma.filled(termz_t, np.nan)
    part1 = (np.exp(-(abs(melt_s - lat_2d_t2)**3))/0.1)
    part2 = (np.exp(-(abs(melt_n - lat_2d_t2)**3))/0.1)
    part3 = (np.exp(-(abs(melt_s + 5 - lat_2d_t2)**1))/0.1)
    part4 = (np.exp(-(abs(melt_n - 5 - lat_2d_t2)**1))/0.1)

    max_1 = np.max(part1)
    if max_1 == 0: max_1 = 1
    max_2 = np.max(part2)
    if max_2 == 0: max_2 = 1
    max_3 = np.max(part3)
    if max_3 == 0: max_3 = 1
    max_4 = np.max(part4)
    if max_4 == 0: max_4 = 1
    
    termy = 100*(part1/max_1 + part2/max_2 + part3/max_3 + part4/max_4) + 1
    Keddy_v3 = termy * termz

    # Choose whether to use plumes:
    if par.convect == True:
        Keddy = Keddy_v3
    else:
        Keddy = Keddy_v1 

    dely_1 = (par.radius + z_vector2[:-1,:])*(par.rlat[1] - par.rlat[0])
    tempd1 = npm.repmat(dely_1, 1,par.max_water_masses)
    dely_1_fit = tempd1.reshape((par.n_b,par.max_water_masses,par.nlat))
    take_in = delz[:-1,:]
    temp_in = npm.repmat(take_in, 1,par.max_water_masses)
    in_delz = temp_in.reshape((par.n_b,par.max_water_masses,par.nlat))
    w_eddy_in = np.zeros([par.n_B, par.n_b, par.max_water_masses, par.nlat])
    w_eddy_in[nB,:,:,:] = (Keddy[nB,:,:,:] / in_delz) * dely_1_fit
    dely_2 = (par.radius + z_vector1[:-1,:])*(par.rlat[1] - par.rlat[0])
    tempd2 = npm.repmat(dely_2, 1,par.max_water_masses)
    dely_2_fit = tempd2.reshape((par.n_b,par.max_water_masses,par.nlat))
    delz = z_vector1 - z_vector2
    take_out = take_in
    take_out[0,:] = take_out[0,:] * 100 # gets removed later
    take_out[1:,:] = delz[:-2,:]
    temp_out = npm.repmat(take_out, 1,par.max_water_masses)
    out_delz = temp_out.reshape((par.n_b,par.max_water_masses,par.nlat))
    w_eddy_out = np.zeros([par.n_B, par.n_b, par.max_water_masses, par.nlat])
    w_eddy_out[nB,1:,:,:] = (Keddy[nB,:-1,:,:] * (1/out_delz[1:,:,:])) 
    w_eddy_out[:,0,:,:] = w_eddy_out[:,0,:,:] * 0 
    w_eddy_out = w_eddy_out * dely_2_fit
    
    w_eddy_net = (np.nan_to_num(w_eddy_in)) - (np.nan_to_num(w_eddy_out))

    width = abs(2*np.pi*par.radius*np.cos(par.rlat))
    width_2 = npm.repmat(width, 1, par.n_b*par.max_water_masses)
    width_3 = width_2.reshape((par.n_B, par.n_b, par.max_water_masses, par.nlat))
    delrlat = par.rlat[1] - par.rlat[0]
    dely = par.radius*(delrlat)        
    phi_eddy_net = w_eddy_net * width_3 

    phi_total = -1*(net_phi_from_sides) - (phi_eddy_net)
    dz_dt = phi_total / (width_3 *dely)

    dz_dt_final = np.where(np.isnan(interfaces), np.nan, dz_dt)

    return dz_dt_final, net_phi_from_sides, (w_eddy_in), (w_eddy_out), phi_bottom_layer, phi_from_sides



def get_slopes(par, interfaces, mx):
    # Part 1: surface slopes at mixed-layer interface.
    surf_slopes = np.zeros([par.n_B, par.n_b, par.max_water_masses, 2])
    surf_flux_out = np.zeros([par.n_B, par.n_b, par.max_water_masses, 2])

    for nB in range(par.n_B):
        for nb in range(par.n_b):
            for nm in range(par.max_water_masses):               
                if np.isnan(mx[nB, nb, nm, 0]) == True or np.isnan(mx[nB, nb, nm, 1]) == True:
                    surf_slopes[nB, nb, nm, 0] = np.nan
                elif mx[nB, nb, nm, 1] < par.lat[0]:
                    surf_slopes[nB, nb, nm, 0] = np.nan
                else: 
                    _, nlat = firstNonNan(interfaces[nB, nb, nm, :], par)
                    ice_to_subtract = (-1*par.d_mixed_layer + par.ice_depth[nlat])
                    z_mean = np.nanmean(interfaces[nB, nb, nm, nlat])
                    del_y = (par.lat[nlat] - mx[nB, nb, nm, 1])*2*np.pi*par.radius/360 
                    if abs(del_y) < (.01): 
                        z_mean = np.nanmean(interfaces[nB, nb, nm, nlat:(nlat+2)])
                        ice_to_subtract = (-1*par.d_mixed_layer + np.nanmean(par.ice_depth[nlat:(nlat+2)]))
                        del_y = (par.lat[nlat] - mx[nB, nb, nm, 1] + (1*(par.lat[2] - par.lat[1])))*2*np.pi*par.radius/360
                    surf_slopes[nB, nb, nm, 0] = (z_mean - (ice_to_subtract))/del_y
                    surf_flux_out[nB, nb, nm, 0] = par.K*(par.slopes_top[nlat] - surf_slopes[nB, nb, nm, 0])*(2*np.pi*par.radius*np.cos(par.rlat[nlat]))
                
                if np.isnan(mx[nB, nb, nm, 2]) == True or np.isnan(mx[nB, nb, nm, 3]) == True:
                    surf_slopes[nB, nb, nm, 1] = np.nan
                elif mx[nB, nb, nm, 3] > par.lat[-1]:
                    surf_slopes[nB, nb, nm, 1] = np.nan
                else:      
                    flip_it = interfaces[nB, nb, nm, :]
                    z_value, nlat_t = firstNonNan(flip_it[::-1], par)
                    nlat = par.nlat - nlat_t - 1
                    outcrop_latitude = mx[0,nb,nm,3] * np.pi/180
                    ice_at_outcrop =  (-10000*(np.cos(outcrop_latitude)**2))
                    ice_to_subtract = (-1*par.d_mixed_layer + ice_at_outcrop)
                    ice_slope_at_outcrop = 10000*(2*np.cos(outcrop_latitude)*np.sin(outcrop_latitude))/par.radius

                    z_mean = np.nanmean(interfaces[nB, nb, nm, nlat])
                    del_y = (mx[nB, nb, nm, 3] - par.lat[nlat])*2*np.pi*par.radius/360
                    if abs(del_y) < (.01): 
                        z_mean = np.nanmean(interfaces[nB, nb, nm, nlat-1:(nlat+1)])
                        ice_to_subtract = (-1*par.d_mixed_layer + 0.5*(par.ice_depth[nlat-1] + ice_at_outcrop))
                        del_y = (mx[nB, nb, nm, 3] - par.lat[nlat] + (1*(par.lat[2] - par.lat[1])))*2*np.pi*par.radius/360
                    surf_slopes[nB, nb, nm, 1] = (ice_to_subtract - z_mean)/del_y
                    surf_flux_out[nB, nb, nm, 1] = -par.K*(ice_slope_at_outcrop - surf_slopes[nB, nb, nm, 1])*(2*np.pi*par.radius*np.cos(outcrop_latitude)) 
   
    # Part 2: Slopes below the surface
    dely = 2*np.pi*par.radius*(par.lat[1] - par.lat[0])/360
    dz_dy = np.zeros([par.n_B, par.n_b, par.max_water_masses, (par.nlat - 1)])
    dz_dy_outcrops = np.zeros([par.n_B, par.n_b, par.max_water_masses, (par.nlat - 1)])
    for nlat in range(par.nlat-1):
        dz_dy[:,:,:,nlat] = (interfaces[:,:,:,nlat+1] - interfaces[:,:,:,nlat])/dely
        dz_dy_outcrops[:,:,:,nlat] = (interfaces[:,:,:,nlat+1] - interfaces[:,:,:,nlat])/dely

    for nB in range(par.n_B):
        for nb in range(par.n_b):
            for nm in range(par.max_water_masses):       
                for nlat in range(par.nlat-1):
                    if np.isnan(dz_dy[nB, nb, nm, nlat]) == True:
                        if np.isnan(interfaces[nB, nb, nm, nlat]) and np.isnan(interfaces[nB, nb, nm, nlat+1]):
                            dz_dy[nB, nb, nm, nlat] =  par.slopes_mid_top[nlat]
                            dz_dy_outcrops[nB, nb, nm, nlat] =  par.slopes_mid_top[nlat]  
                        # now consider cross overs.
                        elif np.isnan(interfaces[nB, nb, nm, nlat]) == True and np.isnan(interfaces[nB, nb, nm, nlat+1]) == False:
                            # south outcrop
                            dz_dy[nB, nb, nm, nlat] = par.slopes_mid_top[nlat] 
                            dz_dy_outcrops[nB, nb, nm, nlat] = par.slopes_mid_top[nlat]
                        elif np.isnan(interfaces[nB, nb, nm, nlat]) == False and np.isnan(interfaces[nB, nb, nm, nlat+1]) == True:
                            # north outcrop
                            dz_dy[nB, nb, nm, nlat] = par.slopes_mid_top[nlat]
                            dz_dy_outcrops[nB, nb, nm, nlat] = par.slopes_mid_top[nlat]
                        else: print('Error: something went wrong with get_slopes')

    return dz_dy, surf_slopes, surf_flux_out, dz_dy_outcrops


def firstNonNan(vector, par):
    """ This function finds the first non-nan term in a vector. 
    It returns the first non-nan value, and it's index.
    [iten, index] = firstNonNan(vector) """    
    nlat = 0
    for item in vector:
        nlat = nlat + 1
        if math.isnan(item) == False:
            return item, (nlat-1)
    raise StopIteration('Error: there was no non nan')

    
def add_time_step_interface(par, var, dvar_dt, dt):
    # Update layers, starting from the top.  
    for nB in range(par.n_B):
        d_total = np.zeros([par.nlat])
        for nb in range(par.n_b):
            for nm in range(par.max_water_masses):
                d_this_layer = np.nan_to_num(dvar_dt[nB, nb, nm, :])*dt
                d_total[:] = d_total[:] + d_this_layer
                var[nB, nb, nm, :] = var[nB, nb, nm, :] + d_total
    return var


def get_delz(par, z_vector1, z_vector2):
    assert par.n_B == 1, "Not setup for multi-basin yet."
    for nb in range(par.n_b):
        for nlat in range(par.nlat):
            z1 = z_vector1[nb, nlat]
            z2 = z_vector2[nb, nlat]
            if np.isnan(z1) == False and z1 != par.ice_depth[nlat] - par.d_mixed_layer:
                if np.isnan(z2):
                    keep_looping = True
                    nb_loop = nb
                    while nb_loop < par.n_b and keep_looping == True:
                        keep_looping = np.isnan(z_vector2[nb_loop, nlat])
                        nb_loop += 1
                    if keep_looping == True:
                        z_vector2[nb, nlat] = par.bottom
                    elif keep_looping == False:
                        z_vector2[nb, nlat] = z_vector2[nb_loop, nlat]
                    else: raise StopIteration('error in delz')
            if (z1 - z_vector2[nb, nlat]) <= 0: 
                z_vector2[nb, nlat] = z1 - .001
            
            if np.isnan(z2) == False and np.isnan(z1) == True:
                keep_looping = True
                nb_loop = nb
                while nb_loop > 0 and keep_looping == True:
                    keep_looping = np.isnan(z_vector1[nb_loop, nlat])
                    nb_loop += -1
                if keep_looping == True:
                    z_vector1[nb, nlat] = par.ice_depth[nlat] - par.d_mixed_layer
                elif keep_looping == False:
                    z_vector1[nb, nlat] = z_vector1[nb_loop, nlat]
                else: raise StopIteration('error in delz')
    return z_vector1, z_vector2



def add_time_step_mixedlayer(par, mixed_layer_data, dyintercept_dt, dt):
    # Update mixedlayer
    take = [1,3]
    take_mix = mixed_layer_data[:,:,:,take]
    take_mix_save = mixed_layer_data[:,:,:,take]
    take_mix[:,:,:,0] = np.where(np.isnan(take_mix[:,:,:,0])==True, -90, take_mix[:,:,:,0])
    take_mix[:,:,:,1] = np.where(np.isnan(take_mix[:,:,:,1])==True, 90, take_mix[:,:,:,1])
    
    for nB in range(par.n_B):
        for nb in range(par.n_b-1,-1,-1):
            for nm in range(par.max_water_masses):
                take_mix_here = take_mix[nB, nb, nm,:]
                vector = relative_position(take_mix, take_mix_here)
                take_mix_next = take_mix_here + dt * np.nan_to_num(dyintercept_dt[nB, nb, nm, :])
                take_mix_edt = take_mix
                take_mix_edt[nB, nb, nm, :] = take_mix_next  
                vector2 = relative_position(take_mix_edt, take_mix_next)                        
                if vector[0] == vector2[0] and vector[2] == vector2[2]: 
                    take_mix[nB, nb, nm, 0] = take_mix_next[0]
                if vector[1] == vector2[1] and vector[3] == vector2[3]: 
                    take_mix[nB, nb, nm, 1] = take_mix_next[1]
    take_mix[:,:,:,0] = np.where(np.isnan(take_mix_save[:,:,:,0])==True, np.nan, take_mix[:,:,:,0])    
    take_mix[:,:,:,1] = np.where(np.isnan(take_mix_save[:,:,:,1])==True, np.nan, take_mix[:,:,:,1])    
    mixed_layer_data[:,:,:,1] = take_mix[:,:,:,0]
    mixed_layer_data[:,:,:,3] = take_mix[:,:,:,1]
    return mixed_layer_data

def relative_position(take_mix, take_mix_here):
    smaller_south = np.sum(take_mix[:,:,:,0] < take_mix_here[0])
    smaller_north = np.sum(take_mix[:,:,:,1] < take_mix_here[1])
    larger_south = np.sum(take_mix[:,:,:,0] > take_mix_here[0])
    larger_north = np.sum(take_mix[:,:,:,1] > take_mix_here[1])
    return [smaller_south, smaller_north, larger_south, larger_north] 
