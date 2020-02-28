#!/usr/bin/python

import numpy as np
import matplotlib as mpl
import glob
import os
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.stats
from scipy import stats
import xarray as xr

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
alphabet = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)',
           '(l)','(m)','(n)','(o)','(p)','(q)','(r)','(s)','(t)','(u)','(v)','(w)','(x)','(y)','(z)',
           '(aa)','(ab)','(ac)','(ad)','(ae)','(af)','(ag)','(ah)','(ai)','(aj)','(ak)','(al)','(am)','(an)',
           '(ao)','(ap)','(aq)','(ar)','(as)','(at)','(au)','(av)','(aw)','(ax)','(ay)','(az)']

def get_lats(ds):
    
        if 'latitude' in ds.coords:
            lat = ds.latitude.values
            lon = ds.longitude.values
        elif 'lat' in ds.coords:
            lat = ds.lat.values
            lon = ds.lon.values
        elif 'nav_lat' in ds.coords:
            lat = ds.nav_lat.values
            lon = ds.nav_lon.values
        lat = np.where(lat>1e20,0.,lat)
        if len(lat.shape)>2:
            lat = lat[0,:,:]
            lon = lon[0,:,:]
        if lat.shape != lon.shape:
            #print(lat)
            #print(lon)
            lat = np.swapaxes(np.tile(lat,(len(lon),1)),0,1)
            lon = np.tile(lon,(len(lat),1))
            
        #print(lat.shape,lon.shape)
        
        return lat, lon
    
def lat_renamer(ds):
    
    if 'latitude' in ds.coords:
        ds = ds
        #print('no action required')
    elif 'lat' in ds.coords: 
        ds = ds.rename({'lat':'latitude'})
        ds = ds.rename({'lon':'longitude'})
    elif 'nav_lat' in ds.coords:
        ds = ds.rename({'nav_lat':'latitude'})
        ds = ds.rename({'nav_lon':'longitude'})
    elif 'rlat' in ds.coords:
        ds = ds.rename({'rlat':'latitude'})
        ds = ds.rename({'rlon':'longitude'})
 
    else:
        print('no option for lats')
        print(ds.coords)
    return ds


def add_trend_stats_to_ds (ds,var,varname): #var = is a data array

    array_names = ['slope', 'intercept','r_value', 'p_value','std_error']
    array_names = [varname+'_'+f for f in array_names]
    ndim = var.shape
    mystats = []
    
    if len(ndim) == 3:
        for i in range(ndim[0]):
            for j in range(ndim[2]):
                ydata = var[i,:,j].values
                xdata = ds.year.values[:len(ydata)]
                slope, intercept, r_value, p_value, std_error = stats.linregress(xdata[~np.isnan(ydata)], ydata[~np.isnan(ydata)])
                mystats.append([slope, intercept, r_value, p_value, std_error])
        mystats = np.asarray(mystats).reshape([ndim[0],ndim[2],5])
        for a, arr_name in enumerate(array_names):
            ds[arr_name] = xr.DataArray(mystats[:,:,a],dims=['name','month'])
  
    elif len(ndim) == 2:
        for i in range(ndim[0]):
                ydata = var[i,:].values
                xdata = ds.year.values
                if np.count_nonzero(np.isnan(ydata))>1:
                    slope, r_value, p_value = np.nan, np.nan, np.nan
                else:
                    slope, intercept, r_value, p_value, std_error = stats.linregress(xdata[~np.isnan(ydata)], ydata[~np.isnan(ydata)])
                mystats.append([slope, intercept, r_value, p_value, std_error])
        mystats = np.asarray(mystats).reshape([ndim[0],5])
        for a, arr_name in enumerate(array_names):
            ds[arr_name] = xr.DataArray(mystats[:,a],dims=['name'])
  
    return ds


def linregress(first_samples, second_samples, dim):
    slope, intercept, r_value, p_value, std_err = xr.apply_ufunc(scipy.stats.linregress,
                       first_samples, second_samples,
                       input_core_dims  = [[dim], [dim]], 
                       output_core_dims = [[],[],[],[],[]],
                       vectorize=True)
        
    return slope, intercept, r_value, p_value, std_err
