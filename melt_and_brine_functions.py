import numpy as np

def get_Fnet(par, mx):
    # basic set up:
    take_location = [1, 3]
    take_tb = [0,2]
    mx_locations = mx[:,:,:,take_location]
    mx_tb = mx[:,:,:,take_tb]
    if np.all(np.isnan(mx_locations)):
        mx_locations[0,0,0,:] = [-90, 90]
        mx_tb[0,0,0,:] = [0, 0]
    distance_melt_source_south_part1 = np.zeros_like(mx_locations)*np.nan

    # melt, top:
    distance_melt_source_south_part1 = (abs(par.y_melt_top[:,:,:,:,0] - mx_locations)* (1 - mx_tb))
    dms1 = np.ma.filled(distance_melt_source_south_part1.astype(float), np.nan)
    distance_melt_source_north_part1 = (abs(par.y_melt_top[:,:,:,:,1] - mx_locations)* (1 - mx_tb))
    dmn1 = np.ma.filled(distance_melt_source_north_part1.astype(float), np.nan)
    # unit correction:
    distance_melt_source_south_part1 = dms1 *2*np.pi*par.radius / 360
    distance_melt_source_north_part1 = dmn1 *2*np.pi*par.radius / 360

    F_melt = par.F_0 * (np.exp(-1*((distance_melt_source_north_part1/par.sig_melt)**2)) 
        + np.exp(-1*((distance_melt_source_south_part1/par.sig_melt)**2)))
    
    # Brine, top:
    distance_brine_source_south_part1 = (abs(par.y_brine_top[:,:,:,:,0] - mx_locations)* (1 - mx_tb))
    dbs1 = np.ma.filled(distance_brine_source_south_part1.astype(float), np.nan)
    distance_brine_source_north_part1 = (abs(par.y_brine_top[:,:,:,:,1] - mx_locations)* (1 - mx_tb))
    dbn1 = np.ma.filled(distance_brine_source_north_part1.astype(float), np.nan)
    # unit correction:
    distance_brine_source_south_part1 = dbs1 *2*np.pi*par.radius / 360
    distance_brine_source_north_part1 = dbn1 *2*np.pi*par.radius / 360

    F_brine = -1*par.gamma*par.F_0 * (np.exp(-1*((distance_brine_source_north_part1/par.sig_brine)**2))
        + np.exp(-1*((distance_brine_source_south_part1/par.sig_brine)**2)))

    F_net = F_melt + F_brine
    return F_net

def find_min_in_3d_function(par, locations, tbs, which_tb):
    tracker = 100
    indexes = np.zeros(3) * np.nan
    for nb in range(par.n_b):
        for nm in range(par.max_water_masses):
            for ns in range(2):
                this_location = locations[nb, nm, ns]
                this_tb = tbs[nb, nm, ns]
                if this_location < tracker and which_tb == this_tb: 
                    tracker = this_location
                    indexes = [nb, nm, ns]
    return indexes
                    
def make_surface_guide(par, mx, phi_res):
    take_locations = [1,3]
    take_tb = [0,2]
    values_here = mx[0,:,:,:]
    locations_here = values_here[:, :, take_locations]
    tb_here = values_here[:, :, take_tb]        
    indexes = np.zeros([3])
    track_indexes = []
    while np.sum(np.isnan(indexes)) == 0:
        indexes = find_min_in_3d_function(par, locations_here, tb_here, 0)
        if np.sum(np.isnan(indexes)) == 0: 
            track_indexes.append(indexes)
            locations_here[indexes[0], indexes[1], indexes[2]] = np.nan
    guide = np.array(track_indexes)
    return guide

def find_nearest_grid_point_south(par, location):
    diff = (par.lat - location)
    diff = np.where(diff < 0, diff, np.nan)
    if np.sum(np.isnan(diff)) == par.nlat:
        where = 0
    else: where = np.nanargmax(diff)
    return where

def find_nearest_grid_point_north(par, location):
    diff = (par.lat - location)
    diff = np.where(diff > 0, diff, np.nan)
    if np.sum(np.isnan(diff)) == par.nlat:
        where = par.nlat - 1
    else: where = np.nanargmin(diff)
    return where    


def get_flow_into_each_layer(par, mx, phi_res, guide, interfaces, Fnet, surf_flux_out, slopes_surf):
    assert phi_res.shape[0] == 1,"Error in get flow."
    buoyancy_fluxes_into_layers = np.zeros_like(interfaces)
    total_fluxes_into_layers = np.zeros_like(interfaces)
    
    # flux at each interface:
    vectorA = np.zeros(guide.shape[0]+1)
    vectorB = np.zeros(guide.shape[0]+1)
    for i in range(guide.shape[0]):
        vectorA[i+1] = phi_res[0, guide[i,0], guide[i,1], guide[i,2]]
        vectorB[i] = phi_res[0, guide[i,0], guide[i,1], guide[i,2]]
    phi_diff = vectorA - vectorB

    count = 0
    for i in range(phi_diff.shape[0]):
        buoyancy_fluxes_into_layers, total_fluxes_into_layers = find_layer_nearest_surface(par, phi_diff, count, guide, buoyancy_fluxes_into_layers, total_fluxes_into_layers, phi_res, interfaces, mx, slopes_surf, surf_flux_out)
        count +=1

    return buoyancy_fluxes_into_layers, total_fluxes_into_layers


def get_melt_and_brine_values(par, mx, interfaces, surf_flux_out, slopes_surf):   
    F_net = get_Fnet(par, mx)
    
    phi_res = F_net * np.nan
    phi_res_strm = F_net * np.nan
    phi_res_for_conv = F_net * np.nan
    
    for nb in range(par.n_b):
        del_y_meter = get_delta_y(par, mx, nb)
        assert mx.shape[0] == 1
        assert mx.shape[2] == 1
        basin_width_s = 2*np.pi*par.radius*np.cos(mx[0,nb,0,1]*np.pi/180)
        basin_width_n = 2*np.pi*par.radius*np.cos(mx[0,nb,0,3]*np.pi/180)
        # south side
        phi_res[:,nb,:,0] = F_net[:,nb,:,0] * del_y_meter[0] * basin_width_s / (par.buoyancy[nb] - par.buoyancy[nb+1])
        phi_res_strm[:,nb,:,0] = F_net[:,nb,:,0] * del_y_meter[0] / (par.buoyancy[nb] - par.buoyancy[nb+1])
        phi_res_for_conv[:,nb,:,0] = F_net[:,nb,:,0] / (par.buoyancy[nb] - par.buoyancy[nb+1])
        # north side
        phi_res[:,nb,:,1] = F_net[:,nb,:,1] * del_y_meter[1] * basin_width_n / (par.buoyancy[nb+1] - par.buoyancy[nb])
        phi_res_strm[:,nb,:,1] = F_net[:,nb,:,1] * del_y_meter[1] / (par.buoyancy[nb+1] - par.buoyancy[nb])
        phi_res_for_conv[:,nb,:,1] = F_net[:,nb,:,1] / (par.buoyancy[nb+1] - par.buoyancy[nb])

    guide = make_surface_guide(par, mx, phi_res) 
    [buoyancy_fluxes_into_layers, total_fluxes_into_layers] = get_flow_into_each_layer(par, mx, phi_res, guide, interfaces, F_net, surf_flux_out, slopes_surf)
    return phi_res, buoyancy_fluxes_into_layers, phi_res_strm, total_fluxes_into_layers


def get_delta_y(par, mx, nb):
    delta_y = np.zeros(2) * np.nan
    poles = [-90,90]
    for side in [0,1]:
        location_index = 2*side + 1
        if np.isnan(mx[0,nb,0,(2*side)]) == True:
            delta_y[side] = np.nan
        # check if it outcrops:
        elif np.isnan(mx[0,nb,0,(location_index)]):
            # if not
            if nb > 0 and np.isnan(mx[0,(nb-1),0,side]) == False:
                delta_y[side] = abs(mx[0,(nb-1),0, location_index] - poles[side])/2
            else:
                delta_y[side] = np.nan
        else:
            # if it outcrops:
            if nb == 0:
                term1 = abs(mx[0,nb,0,3] - mx[0,nb,0,1])/2
                if np.isnan(term1): 
                    term1 = abs(mx[0,nb,0,location_index] - poles[side-1])/2
                term2 = abs(mx[0,nb,0,(location_index)] - mx[0,nb+1,0,(location_index)])/2
                if np.isnan(term2): term2 = abs(mx[0,nb,0,(location_index)] - poles[side])/2
            elif nb == (par.n_b-1):
                term1 = abs(mx[0,nb,0,location_index] - mx[0,nb-1,0,location_index])/2
                if side == 1 and mx[0,nb,0,(location_index)] < 0:
                    term2 = abs(mx[0,nb,0,(location_index)] - 0)/2                
                else:
                    term2 = abs(mx[0,nb,0,(location_index)] - poles[side])/2            
            else:
                term1 = abs(mx[0,nb,0,location_index] - mx[0,nb-1,0,location_index])/2
                term2 = abs(mx[0,nb+1,0,location_index] - mx[0,nb,0,location_index])/2
                if np.isnan(term2): term2 = abs(mx[0,nb,0,(location_index)] - poles[side])/2
            delta_y[side] = term1 + term2
        del_y_meter = 2*np.pi * (delta_y/360) * par.radius
    return del_y_meter



def find_layer_nearest_surface(par, phi_diff, count, guide, b_flux_into_layers, total_fluxes_into_layers, phi_res, interfaces, mx, slopes_surf, surf_flux_out): 
    where_it_converges = np.nan
    take = [1,3]
    mx_take = mx[:,:,:,take]
    mx_clean = mx_take[0,:,:,:]
     
    if count == 0:
        if np.sum(np.isnan(interfaces[0,:,0,0])) == par.n_b:
            # case if there is no non-outcropping layer.
            return b_flux_into_layers, total_fluxes_into_layers
        elif np.isnan(mx_clean[0,0,0]) and np.isnan(interfaces[0,0,0,0]) == False:
            location_next_outcrop = mx_clean[0,0,1]
            location_for_flow = find_nearest_grid_point_south(par, location_next_outcrop) 
            total_fluxes_into_layers[0, 0, 0, location_for_flow] = phi_diff[count]  - surf_flux_out[0,0,0,1] 
            b_flux_into_layers[0, 0, 0, location_for_flow] = phi_diff[count]
            return b_flux_into_layers, total_fluxes_into_layers
        else: 
            n_layers = par.n_b - np.sum(np.isnan(interfaces[0,:,0,0])) + 1
            f_each_layer = phi_diff[count] / n_layers
            nb = 0 
            while np.isnan(interfaces[0,nb,0,0]) == True:
                nb += 1
            where_it_converges = 0
            b_flux_into_layers[0, nb:, 0, where_it_converges] = f_each_layer 
            total_fluxes_into_layers[0, nb:, 0, where_it_converges] = f_each_layer
            return b_flux_into_layers, total_fluxes_into_layers
    
    # similar for the opposite end:
    if count == (phi_diff.shape[0]-1):
        if np.sum(np.isnan(interfaces[0,:,0,-1])) == par.n_b:
            # case if there is no non-outcropping layer.
            return b_flux_into_layers, total_fluxes_into_layers
        else: 
            nb = 0 
            n_layers = par.n_b - np.sum(np.isnan(interfaces[0,:,0,-1])) + 1
            f_each_layer = phi_diff[count] / n_layers
            while np.isnan(interfaces[0,nb,0,-1]) == True:
                nb += 1
            where_it_converges = -1
            b_flux_into_layers[0, nb:, 0, where_it_converges] = f_each_layer 
            total_fluxes_into_layers[0, nb:, 0, where_it_converges] = f_each_layer
            return b_flux_into_layers, total_fluxes_into_layers

    # for other cases: 
    guide_here = guide[count-1,:]
    guide_next = guide[count,:]
    nb = guide_here[0]
    n_s = guide_here[2] # 0 if south, 1 if north
    location_outcrop = mx_clean[nb, guide_here[1], n_s]

    # South Side: 
    if n_s == 0:
        location_for_flow = find_nearest_grid_point_north(par, location_outcrop)
        if np.isnan(interfaces[0,nb,0,location_for_flow-1]) == False and location_for_flow > 0: 
            location_for_flow += -1
    
    if nb == guide_next[0] and n_s == 0:
        location_for_flow = find_nearest_grid_point_north(par, location_outcrop)
        if np.isnan(interfaces[0,nb,0,location_for_flow-1]) == False and location_for_flow > 0: 
            location_for_flow += -1
        b_flux_into_layers[0, nb, 0, location_for_flow] = phi_res[0, nb, 0, 0]  
        total_fluxes_into_layers[0, nb, 0, location_for_flow] = phi_res[0, nb, 0, 0]  - surf_flux_out[0,nb,0,0] 
        
        location_next_outcrop = mx_clean[nb, guide_next[1], 1]
        location_other_flow = find_nearest_grid_point_south(par, location_next_outcrop)
        if np.isnan(interfaces[0,nb,0,location_other_flow+1]) == False and location_other_flow < (par.nlat-1): 
            location_other_flow += 1
        b_flux_into_layers[0, nb, 0, location_other_flow] = 0 - phi_res[0, nb, 0, 1] 
        total_fluxes_into_layers[0, nb, 0, location_other_flow] = 0 - phi_res[0, nb, 0, 1] - surf_flux_out[0,nb,0,1]
        return b_flux_into_layers, total_fluxes_into_layers

    elif n_s == 0:
        b_flux_into_layers[0, nb, 0 , location_for_flow] = phi_diff[count]
        total_fluxes_into_layers[0, nb, 0 , location_for_flow] = phi_diff[count]  - surf_flux_out[0,nb,0,0]
        return b_flux_into_layers, total_fluxes_into_layers

    # North Side: 
    if n_s == 1:
        location_next_outcrop = mx_clean[guide_next[0], guide_next[1], 1]
        location_for_flow = find_nearest_grid_point_south(par, location_next_outcrop) 

        if np.isnan(interfaces[0,nb+1,0,location_for_flow+1]) == False and location_for_flow < (par.nlat-1): 
            location_for_flow += 1
        n_b_flow = nb + 1
        while np.isnan(interfaces[0,n_b_flow,0,location_for_flow]) == True and n_b_flow < par.n_b:
            n_b_flow += 1
        
        if n_b_flow == par.n_b:
            return b_flux_into_layers, total_fluxes_into_layers
        else:
            where_it_converges = location_for_flow
            b_flux_into_layers[0, n_b_flow, 0, where_it_converges] = phi_diff[count]
            total_fluxes_into_layers[0, n_b_flow, 0, where_it_converges] = phi_diff[count] - surf_flux_out[0,nb,0,1]
        return b_flux_into_layers, total_fluxes_into_layers

    pass
