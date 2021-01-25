import numpy as np
import random

def update_intercepts(par, interface_data, mixed_layer_data):    
    mixed_layer_data = update_layers(par, mixed_layer_data, interface_data)          
    interface_data = z_updates_check_edges(par, mixed_layer_data, interface_data)
    return interface_data, mixed_layer_data

def z_updates_check_edges(par, mixed_layer_data, interface_data):
    top_cutoff = par.ice_depth - par.d_mixed_layer
    bottom_cutoff = par.bottom
    for nB in range(par.n_B):
        for nb in range(par.n_b):
            for mass in range(par.max_water_masses):
                if mixed_layer_data[nB, nb, mass, 1] < par.lat[0] or mixed_layer_data[nB, nb, mass, 1] > par.lat[-1]: 
                    mixed_layer_data[nB, nb, mass, 1] = np.nan                    
                if mixed_layer_data[nB, nb, mass, 3] < par.lat[0] or mixed_layer_data[nB, nb, mass, 3] > par.lat[-1]:
                    mixed_layer_data[nB, nb, mass, 3] = np.nan

                if np.isnan(mixed_layer_data[nB, nb, mass, 1]) == False and np.isnan(mixed_layer_data[nB, nb, mass, 3]) == False:
                    if abs(mixed_layer_data[nB, nb, mass, 1] - mixed_layer_data[nB, nb, mass, 3]) < (par.lat[2] - par.lat[1]):
                        mixed_layer_data[nB, nb, mass, :] = np.nan

                if np.isnan(mixed_layer_data[nB, nb, mass, 0]) and np.isnan(mixed_layer_data[nB, nb, mass, 2]):
                    interface_data[nB, nb, mass, :] = interface_data[nB, nb, mass, :] * np.nan
                else:
                    south_edge = mixed_layer_data[nB, nb, mass, 1]
                    north_edge = mixed_layer_data[nB, nb, mass, 3]
                    for l, lat in enumerate(par.lat):
                        if np.isnan(south_edge) == False and lat < south_edge: 
                            interface_data[nB, nb, mass, l] = np.nan
                        elif np.isnan(north_edge) == False and lat > north_edge: 
                            interface_data[nB, nb, mass, l] = np.nan
                        else: 
                            z = interface_data[nB, nb, mass, l]
                            if z >= top_cutoff[l] - .1 and l < par.nlat - 1 and np.isnan(interface_data[nB, nb, mass, l+1]):
                                interface_data[nB, nb, mass, l] = top_cutoff[l] - 0.1
                                top_cutoff[l] += -.1

                            elif z >= top_cutoff[l] - .1 and l > 0 and np.isnan(interface_data[nB, nb, mass, l-1]):
                                interface_data[nB, nb, mass, l] = top_cutoff[l] - 0.1
                                top_cutoff[l] += -.1

                            elif z >= (top_cutoff[l] - .1):
                                interface_data[nB, nb, mass, l] = top_cutoff[l] - 0.1
                                top_cutoff[l] += -0.1

                            elif z < top_cutoff[l]:
                                interface_data[nB, nb, mass, l] = z 
                                top_cutoff[l] = z 
                                
                            elif np.isnan(z) == True:
                                interface_data[nB, nb, mass, l] = top_cutoff[l] - 1
                                top_cutoff[l] += -1
                            if z < bottom_cutoff:
                                interface_data[nB, nb, mass, l] = bottom_cutoff + par.onf
    return interface_data


def update_layers(par, mx, interfaces):
    mx_clean = np.zeros_like(mx)
    mx_clean[:,:,:,1] = np.where(np.isnan(mx[:,:,:,1]), -90, mx[:,:,:,1])
    mx_clean[:,:,:,3] = np.where(np.isnan(mx[:,:,:,3]), -90, mx[:,:,:,3])

    for nb in range(par.n_b):
        if nb > 0:          
            rand = np.random.rand()
            if abs(mx_clean[0,nb,0,1] - mx_clean[0,nb-1,0,1]) < .5 and mx_clean[0,nb,0,1] != -90 and mx_clean[0,nb-1,0,1] != -90:
                mx_clean[0,:nb,0,1] += 0.05
            if abs(mx_clean[0,nb,0,3] - mx_clean[0,nb-1,0,3]) < 0.5 and mx_clean[0,nb,0,3] != 90 and mx_clean[0,nb-1,0,3] != 90:
                mx_clean[0,:nb,0,3] += -rand
                mx_clean[0,nb,0,3] += rand
                
    for nb in range(par.n_b):
        mx_here = mx_clean[0,nb,0,:]
        if nb == 0:
            south_cutoff = mx_here[1]
            north_cutoff = mx_here[3]
        else:                      
            if mx_here[1] < par.lat[0]:
                south_cutoff = -90
            elif mx_here[1] < south_cutoff:
                south_cutoff = mx_here[1]
            elif mx_here[1] >= south_cutoff:
                mx_clean[0,nb,0,1] = south_cutoff - 0.01
                south_cutoff = south_cutoff - 0.01
            
            if mx_here[3] > par.lat[-1]:
                south_cutoff = 90
            elif mx_here[3] > north_cutoff:
                north_cutoff = mx_here[3]
            elif mx_here[3] <= north_cutoff:
                mx_clean[0,nb,0,3] = north_cutoff + 0.01
                north_cutoff = north_cutoff + 0.01

    mx[:,:,:,1] = np.where(mx_clean[:,:,:,1] < par.lat[0], np.nan, mx_clean[:,:,:,1])
    mx[:,:,:,3] = np.where(mx_clean[:,:,:,3] > par.lat[-1], np.nan, mx_clean[:,:,:,3])
    return mx 
        

        

 