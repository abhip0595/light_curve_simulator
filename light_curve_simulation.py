#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from scipy.interpolate import interp1d as interpole
from IPython.display import display, clear_output
import ipywidgets as widgets
import time as sys_time
import warnings
warnings.filterwarnings("ignore")

#Read SED model file
def get_spectra(file,flux_col='g25'):
    f = fits.open(file)  
    specdata = f[1].data 
    f.close() 
    spec = {"wave":specdata["WAVELENGTH"],"flux":specdata[flux_col]}
    my_wave = np.arange(1.5e3,1e4+1)
    inter = interpole(spec["wave"],spec["flux"],kind="quadratic")
    my_flux = inter(my_wave)
    out = {"wave":my_wave,"flux":my_flux}
    return out

def get_vega_flux():
    vega_file = "alpha_lyr_stis_010.fits"
    spec = get_spectra(vega_file,flux_col="FLUX")
    return spec

#Read extinction
def get_ext(file):
    cnames = ["wave","a","g","cos_","c_ext","k_abs","cos_2"]
    ext_data = pd.read_csv(file,delimiter=" ",
                           index_col=False,header=0,
                           names=cnames)
    ext_data["wave"] = ext_data["wave"]*1e4 #Rescaling
    ext_data["sigma"] = ext_data["c_ext"]*5.8e21
    ext = ext_data.copy().drop(columns=["a","g","cos_","k_abs","cos_2"])
    my_wave = np.arange(1.5e3,1e4+1)
    inter = interpole(ext["wave"],ext["sigma"],kind="quadratic")
    my_sigma = inter(my_wave)
    out = {"wave":my_wave,"sigma":my_sigma}
    return out

#Conversion Solar radius to cm
def sradius_to_cm(radius):
    return 6.9634e10*radius

#Conversion parsec to cm
def parsec_to_cm(distance):
    return 3.086e18*distance

#Rescale flux using scale factor
#Default : 1 Solar Radius, 1 parsec
def rescale_flux(spec,radius=1.,distance=1.):
    rspec = spec.copy()
    scale = (sradius_to_cm(radius)/parsec_to_cm(distance))**2
    rspec["flux"] = spec["flux"]*scale
    return rspec

#Add extinction
#Default no reddening i.e E(B-V) = 0
def add_extinction(spec,ext_curv,ebv=0.):
    spec_ext = spec.copy()
    ext = ext_curv.copy()
    spec_ext["flux"] = spec_ext["flux"]*np.exp(-ext["sigma"]*ebv)
    return spec_ext

#Change to Vega Magnitudes
def flux_to_vega(spec,vega):
    df = pd.DataFrame(columns=["wave","mag"])
    df["wave"] = spec["wave"].copy()
    df["mag"] = -2.5*np.log10(spec["flux"]/vega["flux"])
    df = df.interpolate()
    out = {"wave":df["wave"],"mag":df["mag"]}
    return out

#Get magnitude at required wavelength
#Default wavelength 5500A
def get_mag_(spec,wavelength=5500):
    inter = interpole(spec["wave"],spec["mag"],kind="quadratic")
    out = inter(wavelength)
    return out

#Wrapper to get spectra
def sed_wrapper(ck_file="ckp00_6000.fits",ebv=0.,radius=1.,distance=1.):
    spec = get_spectra(ck_file) 
    spec = rescale_flux(spec,radius=radius,distance=distance)
    ext_file = "kext_albedo_WD_MW_3.1_60_D03.txt"
    ext = get_ext(ext_file)
    nspec = add_extinction(spec,ext,ebv=ebv)
    vega = get_vega_flux()
    spec_mag = flux_to_vega(nspec,vega)
    return spec_mag

#Variation function
def tri_var(t,p):
    return 4/p*np.abs(np.mod(t-p/4,p)-p/2)-1
def sine_var(t,p):
    return np.sin(2*np.pi*t/p)
def var(t,p,fun=sine_var):
    return fun(t,p)

#Wrapper to create output
def wrapper_light_curve_plot(pdict):
    
    #Unpack dict
    new_radius = pdict["new_radius"]
    iday = pdict["iday"]
    days = pdict["days"]
    mag = pdict["mag"]
    mag_change = pdict["mag_change"]
    radius_change = pdict["radius_change"]
    radius = pdict["radius"]
    opacity = pdict["opacity"]
    fig = pdict["fig"]
    ax1 = pdict["ax1"]
    ax2 = pdict["ax2"]
    ax3 = pdict["ax3"]
    ax4 = pdict["ax4"]
    
    #Plot everything    
    ax1.set_xlim(0, iday)
    ax1.cla()
    ax1.plot(days, mag)
    ax1.set_title("Modelled Brightness")
    ax1.set_xlabel("Time(days)")
    ax1.set_ylabel("Magnitude")
    
    ax2.cla()
    ax2.set_title("Normalized Radius")
    ax2.set_aspect('auto')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    x = 2*new_radius/radius * np.outer(np.cos(u), np.sin(v))
    y = 2*new_radius/radius * np.outer(np.sin(u), np.sin(v))
    z = 2*new_radius/radius * np.outer(np.ones(np.size(u)), np.cos(v))
    elev = 10.0
    rot = 80.0 / 180 * np.pi
    ax2.set_xlim([-4,+4])
    ax2.set_ylim([-4,+4])
    ax2.set_zlim([-4,+4])
    ax2.plot_surface(x, y, z,  rstride=4, cstride=4, color='r', linewidth=0, alpha=opacity)
    ax2.view_init(elev = 10, azim = 0)

    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.zaxis.set_ticklabels([])
    
    ax3.set_xlim(0, iday)
    ax3.cla()
    ax3.plot(days, mag_change)
    ax3.set_title("Brightness Variation")
    ax3.set_xlabel("Time(days)")
    ax3.set_ylabel("Change in Magnitude")
    
    ax4.set_xlim(0, iday)
    ax4.cla()
    ax4.plot(days, radius_change)
    ax4.set_title("Radius Variation")
    ax4.set_xlabel("Time(days)")
    ax4.set_ylabel("Change in Radius")
    
    display(fig)
    clear_output(wait = True)
    plt.pause(0.005)
        
def simulate_light_curve():   
    global slider 
    
    slider = {}
    slider["radius"] = widgets.IntSlider(value=10, min=0,max=100,step=1, description='Radius (x Solar Radius)')
    slider["dradius"] = widgets.IntSlider(value=12, min=0,max=50,step=1, description='Max Variation:')
    slider["time"] = widgets.IntSlider(value=30, min=3,max=300,step=1, description='Time Period (days):')
    slider["star_distance"] = widgets.IntSlider(value=100, min=1,max=3000,step=100, description='Distance (parsec):')
    slider["ndays"] = widgets.IntSlider(value=100, min=50,max=1000,step=10, description='Ndays (days):')
    
    button = widgets.Button(description='Simulate')
    out = widgets.Output()
            
    display(slider['radius'])
    display(slider['dradius'])
    display(slider['time'])
    display(slider['star_distance'])
    display(slider['ndays'])
    display(button)
    
    def simulate(button):
        radius = slider['radius'].value
        dradius = slider["dradius"].value
        time = slider["time"].value
        star_distance = slider["star_distance"].value
        ndays = slider["ndays"].value
    
        set_ebv = 0.02 # Set this value
        wavelength = 5550 # Set this value
        time_step = 1
        ck_file = "ckp00_6000.fits"
        
        delta_radius = radius*dradius/100.
                #Initialize
        days = np.array([0])
        spec_mag = sed_wrapper(ck_file=ck_file,ebv=set_ebv,
                                radius=radius,distance=star_distance)
        mag = get_mag_(spec_mag,wavelength=wavelength)
        m0 = mag
        days = np.array([0])
        mag_change = np.array([0])
        radius_arr = np.array(radius)
        radius_change = np.array([0])
    
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2,projection='3d')
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
    
        for iday in np.arange(1,ndays,time_step):
                #Some calculations
            ivar = var(iday,time,fun=sine_var)
            new_radius = radius + delta_radius*ivar   
            spec_mag = sed_wrapper(ck_file=ck_file,ebv=set_ebv,radius=new_radius,distance=star_distance)
            now_mag = get_mag_(spec_mag,wavelength=wavelength)
            days = np.append(days,iday)
            mag = np.append(mag,now_mag)
            mag_change = np.append(mag_change,now_mag-m0)
            radius_arr = np.append(radius_arr,new_radius)
            radius_change = np.append(radius_change,new_radius-radius)
            opacity = 0.8-0.1*ivar
        
            pdict = {"new_radius":new_radius,"iday":iday,"days":days,
                    "mag":mag,"mag_change":mag_change,"radius_change":radius_change,
                    "radius":radius,"opacity":opacity,"fig":fig,"ax1":ax1,"ax2":ax2,
                    "ax3":ax3,"ax4":ax4}
            if np.mod(iday,3) == 0:
                wrapper_light_curve_plot(pdict)
    do_once = False   
    while not do_once:
            button.on_click(simulate)
            do_once = True        