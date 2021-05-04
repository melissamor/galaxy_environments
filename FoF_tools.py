# Tools for FoF algorithm & detecting neighbors

import numpy as np
from astropy.io import fits
from astropy.table import Table,Column
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.integrate import quad


# returns the angular separation between 2 points in the sky
# Input: ra, dec of objects in degrees
def angsep(ra1,dec1,ra2,dec2,deg=True):
    if deg:
        ra1 = ra1*np.pi/180.
        ra2 = ra2*np.pi/180.
        dec1 = dec1*np.pi/180.
        dec2 = dec2*np.pi/180.
    # rounds to 14 after decimal to avoid computational math errors
    return np.arccos(np.round(np.sin(dec1)*np.sin(dec2)+np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2),14))*180./np.pi

# determines if two measurements are within errors of each other
def in_error(v1,e1,v2,e2):
    v = np.array([v1,v2])
    e = np.array([e1,e2])

    maxi = np.where(np.max(v)==v)[0][0]
    mini = np.where(np.min(v)==v)[0][0]

    ine = (v[maxi]-e[maxi])-(v[mini]+e[mini])
    
    if ine >= 0.: return False
    else: return True

# defines schecter function
def schechtfunc(M,Ms=-24.2,a=-1.02,phis=1.08e-2):
    return .4*np.log(10)*phis*10**(.4*(a+1)*(Ms-M))*np.exp(-10**(.4*(Ms-M)))

# calculates absolute magnitude from apparent magnitude and redshift
def absmag(m,z):
    dc = cosmo.comoving_distance(z).to(u.pc).value
    return m-5*np.log10(dc*(1+z)/10)

# determines extent of group with list of group ras, decs, and redshifts
def extent(ras,decs,zs):
    group_z = np.mean(zs)
    group_center = (np.mean(ras),np.mean(decs))
    Mpcperdeg = cosmo.kpc_proper_per_arcmin(group_z).value*60./10**3
    rsep = [(Mpcperdeg*angsep(group_center[0],group_center[1],ras[i],decs[i])) for i in range(len(ras))]
    rsepsq = [(Mpcperdeg*angsep(group_center[0],group_center[1],ras[i],decs[i]))**2 for i in range(len(ras))]
    return np.sqrt(np.sum(rsepsq)/2/len(ras)/(1+group_z)**2)

# count number of galaxies within some radius of specified ra, dec
def count_galaxies(c_ra,c_dec,radius,z,ras,decs,zs,zerrs):
    Mpcperdeg = cosmo.kpc_proper_per_arcmin(z).value*60./10**3
    radius_deg = radius/Mpcperdeg
    ras = ras[np.where([in_error(z,0.,zs[i],zerrs[i]) for i in range(len(zs))])[0]]
    decs = decs[np.where([in_error(z,0.,zs[i],zerrs[i]) for i in range(len(zs))])[0]]
    return len(np.where(angsep(c_ra,c_dec,ras,decs) <= radius_deg)[0])



######################################################################
################### MAJOR FUNCTIONS ##################################
######################################################################

# Searches for all galaxies within redshift and radius of specified ra/dec/z

# INPUT
#  neighbors: table of neighbors, loaded from file
#  z: redshift
#  z_err: redshift error
#  ra: central RA
#  dec: central Dec

def within_radius(ra, dec, z, zerr, neighbors, search_rad = 2*u.Mpc, zerr_cutoff=False, in_z = True):
    if in_z:
        if in_z == True:
            # Only include galaxies whose redshifts fall within errors of the bent double
            neighbors = neighbors[np.where([in_error(z,zerr,i['photo_z'],i['photo_zerr']) for i in neighbors])[0]]
        else:
            # include galaxy in slab of size in_z
            neighbors = neighbors[np.where(abs(z-neighbors['photo_z'])<in_z)]

	    

    # Find the value that corresponds to search_rad in arcsec
    kpcperarcmin = cosmo.kpc_proper_per_arcmin(z)
    rad_arcsec = (search_rad/kpcperarcmin).to(u.arcsec).value
    
    # Only select galaxies that fall within search_rad and update neighbors table
    near_sky = np.where(angsep(ra,dec,neighbors['RA'],neighbors['DEC'])*3600<rad_arcsec)[0]
    neighbors = neighbors[near_sky]
    
    # If a redshift error is specified, excludes galaxies with errors that fall above this
    if zerr_cutoff:
        neighbors = neighbors[np.where(neighbors['photo_zerr']<zerr_cutoff)[0]]
        
    return neighbors

# FoF using specified linking length (in kpc)
def absolute_FoF(bd_ra,bd_dec,bd_z,bd_zerr,neighbors,search_rad = 200., zerr_cutoff = False, in_z = True):
    if in_z:
        if in_z == True:
            # Only include galaxies whose redshifts fall within errors of the bent double
            neighbors = neighbors[np.where([in_error(bd_z,bd_zerr,i['photo_z'],i['photo_zerr']) for i in neighbors])[0]]
        else:
            # include galaxy in slab of size in_z
            neighbors = neighbors[np.where(abs(bd_z-neighbors['photo_z'])<in_z)]
    
    # Calculates initial search radius
    init_dist = search_rad/(cosmo.kpc_proper_per_arcmin(bd_z).value*60.)
    
    # Find galaxies that fall within this radius
    group_ni = np.where(angsep(bd_ra,bd_dec,neighbors['RA'],neighbors['DEC']) < init_dist)[0]
    group_ni = list(group_ni)
    i = False
    for n in group_ni:
        i = neighbors[n]
        
        # Calculate new search distance from selected neighbor
        sdist = search_rad/(cosmo.kpc_proper_per_arcmin((i['photo_z']+bd_z)/2).value*60.)
        
        # Find neighbors of this neighbor
        i_neighbors = np.where(angsep(i['RA'],i['DEC'],neighbors['RA'],neighbors['DEC']) < sdist)[0]
        
        # Loops through all new neighbors and adds them to list of neighbors to loop through if they aren't already in that list
        for j in i_neighbors:
            if j not in group_ni:
                group_ni.append(j)
    if zerr_cutoff:
        neighbors = neighbors[group_ni][np.where(neighbors['photo_zerr'][group_ni]<zerr_cutoff)[0]]
    else:
        neighbors = neighbors[group_ni]
    return neighbors,group_ni


# Version of FoF algorithm that calculates the linking length using the density as an input
	# density calculated from schecter function
def scaled_FoF(bd_ra,bd_dec,bd_z,bd_zerr,neighbors,density,b = 0.11, zerr_cutoff = False, in_z = True):
    if in_z:
        if in_z == True:
            # Only include galaxies whose redshifts fall within errors of the bent double
            neighbors = neighbors[np.where([in_error(bd_z,bd_zerr,i['photo_z'],i['photo_zerr']) for i in neighbors])[0]]
        else:
            # include galaxy in slab of size in_z
            neighbors = neighbors[np.where(abs(bd_z-neighbors['photo_z'])<in_z)]
    
    # Calculates the linking length in kpc (assumes density is in Mpc^-2)
    search_rad = b*density**(-1./3.)*1000
    print (search_rad)
    
    # Calculates initial search radius
    init_dist = search_rad/(cosmo.kpc_proper_per_arcmin(bd_z).value*60.)
    # Find galaxies that fall within this radius
    group_ni = np.where(angsep(bd_ra,bd_dec,neighbors['RA'],neighbors['DEC']) < init_dist)[0]
    group_ni = list(group_ni)
    i = False
    for n in group_ni:
        i = neighbors[n]
        
        # Calculate new search distance from selected neighbor
        sdist = search_rad/(cosmo.kpc_proper_per_arcmin((i['photo_z']+bd_z)/2).value*60.)
        
        # Find neighbors of this neighbor
        i_neighbors = np.where(angsep(i['RA'],i['DEC'],neighbors['RA'],neighbors['DEC']) < sdist)[0]
        
        # Loops through all new neighbors and adds them to list of neighbors to loop through if they aren't already in that list
        for j in i_neighbors:
            if j not in group_ni:
                group_ni.append(j)
    if zerr_cutoff:
        neighbors = neighbors[group_ni][np.where(neighbors['photo_zerr'][group_ni]<zerr_cutoff)[0]]
    else:
        neighbors = neighbors[group_ni]
    return neighbors,group_ni

# Version of FoF algorithm that includes probability-based searching for galaxies in redshift space
def absolute_FoF_prob(
        bd_ra,bd_dec,bd_z,bd_zerr,neighbors,
        search_rad = 200., zerr_cutoff = False, least_prob = .67, zdif = .1, cutoff = 5000.):
    '''
    FoF algorithm that applies probability-based solution in redshift space
    
    search_rad: linking length, in kpc
    zerr_cutoff: maximum allowed redshift error
    least_prob: lowest allowed probability for two galaxies to be at same z
    '''
    # cuts neighbors with z difference larger than zdif
    neighbors = neighbors[np.where(abs(bd_z-neighbors['photo_z'])<zdif)]
    
    # implements redshift error cutoff if specified
    if zerr_cutoff:
        neighbors = neighbors[np.where(neighbors['photo_zerr']<zerr_cutoff)[0]]
    
    # Calculates initial search radius
    init_dist = search_rad/(cosmo.kpc_proper_per_arcmin(bd_z).value*60.)
    
    # Find galaxies that fall within this radius of bent double
    group_ni = np.where(angsep(bd_ra,bd_dec,neighbors['RA'],neighbors['DEC']) < init_dist)[0]
    group_ni = list(group_ni)
    
    max_dist = cutoff/(cosmo.kpc_proper_per_arcmin(bd_z).value*60.)
    
    # loops through indices of galaxies within search radius of bent double
    i = False
    for n in group_ni:
        i = neighbors[n]
        
        # Calculate new search distance from selected neighbor
        sdist = search_rad/(cosmo.kpc_proper_per_arcmin((i['photo_z']+bd_z)/2).value*60.)
        
        if cutoff:
            i_neighbors = np.where((angsep(i['RA'],i['DEC'],neighbors['RA'],neighbors['DEC']) < sdist)&
                                  (angsep(bd_ra,bd_dec,neighbors['RA'],neighbors['DEC']) < max_dist))[0]
        else:
            # Find neighbors of this neighbor
            i_neighbors = np.where(angsep(i['RA'],i['DEC'],neighbors['RA'],neighbors['DEC']) < sdist)[0]
        
        # Loops through all new neighbors and adds them to list of neighbors to loop through if they aren't already in that list
        for j in i_neighbors:
            if j not in group_ni:
                group_ni.append(j)
    
    neighbors = neighbors[group_ni]
    
    # implement probability selection on remaining neighbors list
    gi = []
    for i in range(len(neighbors)):
        # Save the redshifts and errors to 2 different arrays
        vs = np.array([bd_z,neighbors[i]['photo_z']])
        es = np.array([bd_zerr,neighbors[i]['photo_zerr']])

        # Create array of possible start and end points in plot
        pos = [vs[0]-es[0]*4,vs[0]+es[0]*4,vs[1]+es[1]*4,vs[1]-es[1]*4]

        # Create integral array using min and max of previous list
        x = np.linspace(min(pos),max(pos),10000)

        # Create gaussian distributions
        g1 = norm.pdf(x,vs[0],es[0])
        g2 = norm.pdf(x,vs[1],es[1])

        # Define max probability to normalize by
        max_p = g1*norm.pdf(x,vs[0],es[1])
        max_integ = np.trapz(max_p,x)

        # Calculate normalized probability
        prob = np.trapz(g1*g2,x)/max_integ

        if prob > least_prob:
            gi.append(i)
    neighbors = neighbors[gi]
    
    # final table of neighbors
    return neighbors

def within_radius_prob(
        ra, dec, z, zerr, neighbors,
        search_rad = 2*u.Mpc, zerr_cutoff=False, least_prob=.67, zdif = .1):
    '''
    Finds all galaxies within search_rad Mpc of bent double
    
    search_rad: search radius, in Mpc
    zerr_cutoff: maximum allowed redshift error
    least_prob: lowest allowed probability for two galaxies to be at same z
    '''
    # cuts neighbors with z difference larger than 1
    neighbors = neighbors[np.where(abs(z-neighbors['photo_z'])<zdif)]

    # Find the value that corresponds to search_rad in arcsec
    kpcperarcmin = cosmo.kpc_proper_per_arcmin(z)
    rad_arcsec = (search_rad/kpcperarcmin).to(u.arcsec).value
    
    # Only select galaxies that fall within search_rad and update neighbors table
    near_sky = np.where(angsep(ra,dec,neighbors['RA'],neighbors['DEC'])*3600<rad_arcsec)[0]
    neighbors = neighbors[near_sky]
    
    # If a redshift error is specified, excludes galaxies with errors that fall above this
    if zerr_cutoff:
        neighbors = neighbors[np.where(neighbors['photo_zerr']<zerr_cutoff)[0]]
        
    gi = []
    for i in range(len(neighbors)):
        # Save the redshifts and errors to 2 different arrays
        vs = np.array([z,neighbors[i]['photo_z']])
        es = np.array([zerr,neighbors[i]['photo_zerr']])

        # Create array of possible start and end points in plot
        pos = [vs[0]-es[0]*4,vs[0]+es[0]*4,vs[1]+es[1]*4,vs[1]-es[1]*4]

        # Create integral array using min and max of previous list
        x = np.linspace(min(pos),max(pos),10000)

        # Create gaussian distributions
        g1 = norm.pdf(x,vs[0],es[0])
        g2 = norm.pdf(x,vs[1],es[1])

        # Define max probability to normalize by
        max_p = g1*norm.pdf(x,vs[0],es[1])
        max_integ = np.trapz(max_p,x)

        # Calculate normalized probability
        prob = np.trapz(g1*g2,x)/max_integ

        if prob > least_prob:
            gi.append(i)
    neighbors = neighbors[gi]
    
    return neighbors

def get_bd_info(bd_name,bd_info,vis_coords,r=False):
    '''
    Load bent double redshift information
    '''
    # Load coordinates of optical counterpart
    bd_vloc = np.where(vis_coords['name']==bd_name)[0]
    bd_ra = vis_coords['vis_ra'][bd_vloc].data[0]
    bd_dec = vis_coords['vis_dec'][bd_vloc].data[0]
    
    # Load redshift (either spectroscopic or photometric)
    bd_loc = np.where(bd_info['Name']==bd_name)[0]
    if bd_info['specz'][bd_loc] != 'null' and float(bd_info['specz'][bd_loc].data[0]) > 0.:
        bd_z = float(bd_info['specz'][bd_loc].data[0])
        bd_zerr = float(bd_info['speczerr'][bd_loc].data[0])
    
    # Return r-band magnitude, if requested
    if r:
        return bd_ra,bd_dec,bd_z,bd_zerr,bd_info['r'][bd_loc].data[0]
    else:
        return bd_ra,bd_dec,bd_z,bd_zerr