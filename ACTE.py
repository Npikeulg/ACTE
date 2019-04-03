#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Nicholas Pike
Email : Nicholas.pike@smn.uio.no

Purpose: Calculation of the thermal expansion coefficients 
         
Notes: - Oct 4th start of program 
       - Apr 5th two dimensional fit for free energy
       - Nov 21th three dimensional fit for free energy 
       - Jan 29th minor bug fixes and simplification of build_cell routine
       
"""
#import needed functions
import os
import sys
import subprocess
import linecache
import numpy as np
import scipy.optimize as so
import scipy.constants as sc
from mendeleev import element

#define unit relationships
ang_to_m     = sc.physical_constants['Angstrom star'][0] #angstrom to meter
amu_kg       = sc.atomic_mass
J_to_cal     = 1.0/sc.calorie
h            = sc.physical_constants['Planck constant'][0]
kb           = sc.physical_constants['Boltzmann constant'][0]
Na           = sc.physical_constants['Avogadro constant'][0]
kbar_to_GPa  = 0.1
m_to_cm      = 100.0
kgm3_to_cm3  = 1000.0 #density conversion
g_to_kg      = 1000.0
cal_to_J     = 4.184
difftol      = 1E-5

"""
###############################################################################
The following information should be modified by the user to suit their own 
supercomputer. 
###############################################################################
"""
supercomputer_software = 'slurm' #'pbs'
use_scratch            = 'yes'  # yes for stallo
account_number         = 'nn2615k'
account_email          = 'Nicholas.pike@smn.uio.no' 
__root__               = os.getcwd()
    
"""
##############################################################################
VASP parameters. Make sure you change convergence parameters before doing your
calculation
##############################################################################
"""
ecut     = '500'  #value in eV
ediff    = '1E-7' #value in eV   
kdensity = 5        

"""
##############################################################################
TDEP parameters. These parameters may need to be converged.
##############################################################################
"""
natom_ss  = '200'       #number of atoms in the supercell (metals ~100, semiconductor ~200])
rc_cut    = '100'       #second order cut-off radius (100 defaults to the maximum radius)
tmin      = '0'         #minimum temperature
tmax      = '3000'      #maximum temperature
tsteps    = '1500'      #number of temperature steps
qgrid     = '30 30 30'  #q point grid density for DOS
iter_type = '3'         #method for numerical integration
n_configs = '12'        #number of configurations
t_configs = '0.8'       #temperature of configurations as a fraction of Debye temperature
                        # this should be less than one

"""
##############################################################################
Paths to executables and main file name
##############################################################################
"""
########## ABEL ######################
VASPSR   = 'source /usit/abel/u1/espenfl/vasp/bin/.jobfile_local'  #source path to VASP executable
TDEPSR   = '/usit/abel/u1/nicholasp/CODES/TDEP/tdep-devel/bin/'  #source path to TDEP bin of executables
PYTHMOD  = 'module load Anaconda3/5.1.0\nmodule load hdf5/1.8.14_intel\nmodule load intel-libs/2016.2\nmodule load intelmpi.intel/5.0.2\nmodule unload intel/2015.1\nmodule load ifort/2017.1.132-GCC-5.4.0-2.26' #module for python

########## STALLO ####################
#VASPSR   = 'source /home/espenfl/vasp/bin/.jobfile_local'  #source path to VASP executable
#TDEPSR   = '~nicholasp/bin/TDEP/bin/'  #source path to TDEP bin of executables
#PYTHMOD  = 'module load StdEnv\nmodule load intel/2016b\nmodule load HDF5/1.8.17-intel-2016b\nmodule load Python/2.7.12-intel-2016b\n' #module for python

"""
Begin modules used in this program
"""
def main_thermal(DFT_INPUT,tags):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Calculate the thermal expansion coefficients of the material.
    
    Return: None
    """
    if sys.version_info<=(2,7,0):
        print('ERROR: This program requires Python version 2.7.0 or greater.' )
        sys.exit()
          
    # get data from TDEP and DFT calculations
    cell_data = READ_INPUT_DFT(DFT_INPUT)
               
    #calculation of lattice expansion coefficients.        
    print('Launching calculation of  the coefficients of thermal expansion.')
    #launch calculation of linear expansion coefficients
    cell_data = linear_exp(cell_data,tags)
    print('Calculation of the coefficients of thermal expansion are complete\n')    
    
    return None

def READ_INPUT_DFT(DFT_INPUT):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@sintef.no
    
    Purpose: To read an input file which is formatted and contains the results of 
             our DFT, DFPT, and TDEP calculations
    
    Input: DFT_INPUT  - name of formatted input file
    
    OUTPUT: cell_data  - array of data about the unit cell
            
    """
    #store data in arrays
    cell_data = [0] *30
    #open file and look for data 
    print('Reading input data from DFT and DFPT calculations in %s\n' %DFT_INPUT)
    with open(DFT_INPUT,'r') as f:
        for num,line in enumerate(f,1):
            if line.startswith( 'volume'):
                l = line.strip('\n').split(' ')
                cell_data[0] = float(l[1])*ang_to_m**3.0 #cell volume converted to m^3
            elif line.startswith('alat'):
                l = line.strip('\n').split(' ')
                cell_data[1] = float(l[1])*ang_to_m
            elif line.startswith('blat'):
                l = line.strip('\n').split(' ')
                cell_data[2] = float(l[1])*ang_to_m
            elif line.startswith('clat'):
                l = line.strip('\n').split(' ')
                cell_data[3] = float(l[1])*ang_to_m
            elif line.startswith('natom'):
                l = line.strip('\n').split(' ')
                cell_data[4] = int(l[1])    
            elif line.startswith('atpos'):
                l = line.strip('\n').split(' ')
                cell_data[5] = []
                for i in range(1,len(l)):
                    elname = element(l[i])
                    cell_data[5] = np.append(cell_data[5],[l[i],float(elname.mass*amu_kg)])
            
            elif line.startswith('Free energy on grid filenames'):
                #need to read in the next 25 filenames
                files = []
                diff_volumes,speccell = determine_volumes()
                for y in range(int(diff_volumes)):
                    with open(DFT_INPUT,'r') as h:
                        for j,line2 in enumerate(h):
                            l = line2.strip('\n').split(' ')
                            if j ==  num+y:
                                files = np.append(files,l[0])
                cell_data[19] = files
                        
            elif line.startswith('TMIN'):
                l = line.strip('\n').split(' ')
                cell_data[23] = l[1]
            elif line.startswith('TMAX'):
                l = line.strip('\n').split(' ')
                cell_data[24] = l[1]  
            elif line.startswith('TSTEP'):
                l = line.strip('\n').split(' ')
                cell_data[25] = l[1]  
            elif line.startswith('Bulk Mod'):
                l = line.strip('\n').split(' ')
                cell_data[27] = float(l[2])
                  
    #print data that is read in so far to the terminal
    print('   Printing data from DFT and DFPT calculations...\n')
    print('   Input data for the unit cell:')
    print('   alat: \t\t %s meters' %cell_data[1])
    print('   blat: \t\t %s meters' %cell_data[2])
    print('   clat: \t\t %s meters' %cell_data[3])
    print('   volume: \t\t %s meters^3' %cell_data[0])
    print('   natom: \t\t %s'%cell_data[4])
    print('   Tmin: \t\t %s' %cell_data[23])
    print('   Tmax: \t\t %s' %cell_data[24])
    print('   Tstep: \t\t %s\n' %cell_data[25])    
    attype = ''
    for i in range(len(cell_data[5])):
        if i %2 == 0:
            attype += cell_data[5][i]+' '
    print('   atom type: \t %s\n'%attype)
       
    print('Data read in from DFT, DFPT, and TDEP calculations.\n')
    
    return cell_data

def linear_exp(cell_data,tags):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Calculates the linear expansion coefficient using a spline interpolation at
             each temperature step
    
    Output: modified cell_data with linear expansion coefficients
    """
    #split tags
    withbounds = tags[0]
    #withsolver = tags[1]
    #withBEC    = tags[2]
    withpoly    = tags[3]
    
    #initialize variables
    a0              = cell_data[1]/ang_to_m
    b0              = cell_data[2]/ang_to_m
    c0              = cell_data[3]/ang_to_m
    mass            = cell_data[5]
  
    #quick calculations of variables
    total_mass = 0.0
    for i in range(1,len(mass),2):
        total_mass += float(mass[i])
        
    #optimize calculations based on the number of unique lattice parameters 
    """
    Calculations of isotropic systems!
    """
    if np.abs(a0-b0) <= difftol and np.abs(a0-c0) <= difftol:
        volnum       = 6
        num_unique   = 1
        print('   Number of volumes: %s'%volnum)
        print('   Number of unique axes: %i' %num_unique)
        #step 1: read in free energy files
        latt_array,vol_array,engy_array,free_array,cell_data = read_free_energies(volnum,num_unique,cell_data)
               
        #step 2: determine the lattice parameters that minimize the temperature
        latt_data,cell_data = minimize_free(volnum,num_unique,cell_data,latt_array,engy_array,free_array,withbounds)
        
        #step 3: Fit the equation of state
        bulkT,cell_data = fit_EOS(volnum,num_unique,cell_data,engy_array,free_array,vol_array,withbounds)
        
        #step 4: Calculate coefficients of thermal expansion
        lattder,cell_data = find_CTE(volnum,num_unique,cell_data,latt_data,withpoly)
        
        #step 5: Calculate specific heat at constant pressure
        sheat,cell_data = find_CP(volnum,num_unique,cell_data,lattder,bulkT,total_mass)
        
        #step 6: print all calculations to a file
        print_all(volnum,num_unique,cell_data,free_array,latt_data,lattder,bulkT,sheat)
            
                
        """
        Anisotropic system with two similiar axis (hexgonal first with c as the free axis)
        """
    elif np.abs(a0-b0)<= difftol and np.abs(a0-c0) >difftol :
        volnum       = 36
        num_unique   = 2
        print('   Number of volumes: %s'%volnum)
        print('   Number of unique axes: %i' %num_unique)
        #step 1: read in free energy files
        latt_array,vol_array,engy_array,free_array,cell_data = read_free_energies(volnum,num_unique,cell_data)
        
        #step 2: determine the lattice parameters that minimize the temperature
        latt_data,cell_data = minimize_free(volnum,num_unique,cell_data,latt_array,engy_array,free_array,withbounds)
                       
        #step 3: Fit the equation of state
        bulkT,cell_data = fit_EOS(volnum,num_unique,cell_data,engy_array,free_array,vol_array,withbounds)
            
        #step 4: Calculate coefficients of thermal expansion
        lattder, cell_data = find_CTE(volnum,num_unique,cell_data,latt_data,withpoly)

        #step 5: Calculate specific heat at constant pressure
        sheat,cell_data = find_CP(volnum,num_unique,cell_data,lattder,bulkT,total_mass)
                
        #step 6: print all calculations to a file
        print_all(volnum,num_unique,cell_data,free_array,latt_data,lattder,bulkT,sheat)
        
        """
        Fully anisotropic system
        """
    elif np.abs(a0-b0)> difftol and np.abs(a0-c0) >difftol :
        volnum       = 216
        num_unique   = 3
        print('   Number of volumes: %s'%volnum)
        print('   Number of unique axes: %i' %num_unique)
        #step 1: read in free energy files
        latt_array,vol_array,engy_array,free_array,cell_data = read_free_energies(volnum,num_unique,cell_data)
        
        #step 2: determine the lattice parameters that minimize the temperature
        latt_data,cell_data = minimize_free(volnum,num_unique,cell_data,latt_array,engy_array,free_array,withbounds)
                       
        #step 3: Fit the equation of state
        bulkT,cell_data = fit_EOS(volnum,num_unique,cell_data,engy_array,free_array,vol_array,withbounds)
            
        #step 4: Calculate coefficients of thermal expansion
        lattder, cell_data = find_CTE(volnum,num_unique,cell_data,latt_data,withpoly)

        #step 5: Calculate specific heat at constant pressure
        sheat,cell_data = find_CP(volnum,num_unique,cell_data,lattder,bulkT,total_mass)
                
        #step 6: print all calculations to a file
        print_all(volnum,num_unique,cell_data,free_array,latt_data,lattder,bulkT,sheat)
        
    return cell_data

def read_free_energies(volnum,num_unique,cell_data):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@smn.uio.no
    
    Purpose: Read in the free energy files and process them
    
    Return: Arrays of the lattice parameters, volume, electronic energy, free energy, and cell data
    """
    #build arrays
    free_array      = np.empty(shape=(int(volnum),int(cell_data[25])))
    latt_array      = np.empty(shape=(int(volnum),3))
    engy_array      = np.empty(shape=(int(volnum),1))
    vol_array       = []

    #gather necessary data
    files           = cell_data[19]
    
    print('   1 - Reading in the free energy files for each lattice grid')
    found_unstable = False
    unstable_files = []
    unstable_numbs = []
    if num_unique == 1:
        i = 0
        for file in files:
            #use the file name to determine the lattice constants
            l = file.split('_')
            latt_array[i][0] = float(l[3]) # a lattice
            engy_array[i][0] = float(l[6]) # U_0 (a)
            vol_array        = np.append(vol_array,float(l[7])) #volume
            
            try: 
                os.path.isfile('free_energies/'+file)
                
            except:
                print('ERROR: The file %s was not found in the directory or contains errors.' %file)
                sys.exit()    
                
            with open('free_energies/'+file,'r') as f:
                for num,line in enumerate(f,0):
                    l = line.strip('\n').split()
                    if l[1] != 'NaN':
                        free_array[i][num] = float(l[1])
                    if float(l[1]) > 3.0E8:
                        found_unstable = True

            if found_unstable == True:
                print('   ERROR: Free energy calculation for %i may have an instability. Will relaunch at end.'%i)
                print('   Suggestion: Plot the dispersion relation to view the instability.')
                unstable_files = np.append(unstable_files,file)
                unstable_numbs = np.append(unstable_numbs,i)   
                found_unstable = False
                
            i+=1
                
    elif num_unique == 2:
        i = 0
        for file in files:
            #use the file name to determine the lattice constants
            l = file.split('_')
            latt_array[i][0] = float(l[3]) # a lattice
            latt_array[i][2] = float(l[5]) # c lattice
            engy_array[i][0] = float(l[6]) # U_0 (a)
            vol_array        = np.append(vol_array,float(l[7])) #volume
            
            try: 
                os.path.isfile('free_energies/'+file)
                
            except:
                print('ERROR: The file %s was not found in the directory or contains errors.' %file)
                sys.exit()    
                
            with open('free_energies/'+file,'r') as f:
                for num,line in enumerate(f,0):
                    l = line.strip('\n').split()
                    if l[1] != 'NaN':
                        free_array[i][num] = float(l[1])
                    if float(l[1]) > 3.0E8:
                        found_unstable = True
                        
            if found_unstable == True:
                print('   ERROR: Free energy calculation for %i may have an instability. Will relaunch at end.'%i)
                print('   Suggestion: Plot the dispersion relation to view the instability.')
                unstable_files = np.append(unstable_files,file)
                unstable_numbs = np.append(unstable_numbs,i) 
                found_unstable = False

            i+=1
            
    elif num_unique == 3:
        i = 0
        for file in files:
            #use the file name to determine the lattice constants
            l = file.split('_')
            latt_array[i][0] = float(l[3]) # a lattice
            latt_array[i][1] = float(l[4]) # b lattice
            latt_array[i][2] = float(l[5]) # c lattice
            engy_array[i][0] = float(l[6]) # U_0 (a)
            vol_array        = np.append(vol_array,float(l[7])) #volume
            
            try: 
                os.path.isfile('free_energies/'+file)
                
            except:
                print('ERROR: The file %s was not found in the directory or contains errors.' %file)
                sys.exit()    
                
            with open('free_energies/'+file,'r') as f:
                for num,line in enumerate(f,0):
                    l = line.strip('\n').split()
                    if l[1] != 'NaN':
                        free_array[i][num] = float(l[1])
                    if float(l[1]) > 3.0E8:
                        found_unstable = True
                        
            if found_unstable == True:
                print('   ERROR: Free energy calculation for %i may have an instability. Will relaunch at end.'%i)
                print('   Suggestion: Plot the dispersion relation to view the instability.')
                unstable_files = np.append(unstable_files,file)
                unstable_numbs = np.append(unstable_numbs,i) 
                found_unstable = False

            i+=1            
            
    if unstable_numbs !=[]:
        relaunch_configs(unstable_numbs,unstable_files)
        sys.exit()
                              
    return latt_array,vol_array,engy_array,free_array,cell_data

def minimize_free(volnum,num_unique,cell_data,latt_array,engy_array,free_array,withbounds):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Find lattice parameters that minimize the free energy
    
    Return: Arrays of lattice parameters and cell data
    """
    #gather necessary information
    a0              = cell_data[1]/ang_to_m
    b0              = cell_data[2]/ang_to_m
    c0              = cell_data[3]/ang_to_m
    numatoms        = float(cell_data[4]) 
    Tmin            = int(cell_data[23])
    Tmax            = int(cell_data[24])
    tempsteps       = int(cell_data[25])
    desired_acc     = 1.0E-3

    #build arrays
    x         = []
    y         = []
    t         = []
    latt_data = np.empty(shape=(4,tempsteps))
    
    print('   2 - Generating the fits for each temperature and \n'\
          '       finding the minimum set of coordinates.')
    print('       ---Brief pauses are normal---')
    
    with open('out.coefficients','w') as f:
        f.write('# Output coefficients of fit to free energy polynomial.\n')
        f.write('# tstep  cof[0] cof[1] ... etc\n')
    
    if withbounds == True:
        print('       Bounds are being used.')
            
    if num_unique == 1:
        for i in range(int(volnum)):
            x = np.append(x,latt_array[i][0])
                
        #find max and min of the lattice x
        xmin = min(x)
        xmax = max(x)
                                   
        #generate list of coordinates for each temperature   
        for i in range(tempsteps):
            z         = []
            for j in range(int(volnum)):
                z = np.append(z,(engy_array[j][0]+numatoms*free_array[j][i]))  #Add internal energy to each point.

            #internal test that they are all the same length
            try:
                x.shape[0] == z.shape[0]
            except:
                print('ERROR: The dimensions of either x or z are unequal.')
                print('The dimensions of the arrays are %s %s.'%(x.shape[0],z.shape[0]))
                sys.exit()
            
            #find best fit 6th order fit 
            #this is written in the same order as the coefficients of the fit. I.e. c00 , c10, c20 etc.
            A = np.c_[x*0.0+1.0,x, x**2,x**3,x**4]

            #C contains the coefficients to the polynomial we are trying to fit. 
            # if statement added to handle bad behavior in older versions of numpy
            try: 
                C,_,_,_ = np.linalg.lstsq(A, z,rcond=-1)
                 
            except FutureWarning:
                C,_,_,_ = np.linalg.lstsq(A, z,rcond=-1) 
                                                  
            #print the coefficients, C, to a file as a function of temperature
            with open('out.coefficients','a') as f:
                f.write('%s %s %s %s %s %s\n' %(i,C[0],C[1],C[2],C[3],C[4]))
                                
            #use the BFGS method to determine the point that minimizes the function
            if i == 0:
                initial_guess = [a0]
            else:
                initial_guess = [latt_data[1][i-1]]
                
            tol = 1E-10
            if withbounds == True:
                bounds        = [(xmin,xmax),] #bounds of free energy grid
            
                result = so.minimize(fourthorder_ploy,initial_guess,
                                method='L-BFGS-B',args=(C,),bounds=bounds,tol=tol)
            else:
                result = so.minimize(fourthorder_ploy,initial_guess,
                                method='BFGS',args=(C,),tol=tol)
                    
            #gather results from optimization routine
            latt_data[0][i] = Tmin+(Tmax-Tmin)/tempsteps*i
            latt_data[1][i] = result.x[0]
            
    elif num_unique == 2:
        for i in range(int(volnum)):
            x = np.append(x,latt_array[i][0])
            y = np.append(y,latt_array[i][2]) #here y is the c direction   
            
        #find max and min of the lattice parameters
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
               
        #generate list of coordinates for each temperature   
        for i in range(tempsteps):
            z = []
            for j in range(int(volnum)):
                z = np.append(z,(engy_array[j][0]+numatoms*free_array[j][i]))  #Add internal energy to each point.
                        
            #internal test that they are all the same length
            try:
                x.shape[0] == y.shape[0] == z.shape[0]
            except:
                print('ERROR: The dimensions of either x,y, or z are unequal.')
                print('The dimensions of the arrays are %s %s %s.'%(x.shape[0],y.shape[0],z.shape[0]))
                sys.exit()

            #find best fit 6th order fit             
            A = np.c_[x**0*y**0+1, x**1*y**0, x**2*y**0, x**3*y**0, x**4*y**0,
                      x**0*y**1, x**1*y**1, x**2*y**1, x**3*y**1, x**4*y**1,
                      x**0*y**2, x**1*y**2, x**2*y**2, x**3*y**2, x**4*y**2,
                      x**0*y**3, x**1*y**3, x**2*y**3, x**3*y**3, x**4*y**3,
                      x**0*y**4, x**1*y**4, x**2*y**4, x**3*y**4, x**4*y**4]
            
            #C contains the coefficients to the polynomial we are trying to fit. 
            # if statement added to handle bad behavior in older versions of numpy
            try: 
                C,_,_,_ = np.linalg.lstsq(A, z,rcond=-1)
                 
            except FutureWarning:
                C,_,_,_ = np.linalg.lstsq(A, z,rcond=-1) 
            
            #print the coefficients, C, to a file as a function of temperature
            with open('out.coefficients','a') as f:
                f.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s '\
                        '%s %s %s %s %s\n' 
                        %(i,C[0],C[1],C[2],C[3],C[4],C[5],C[6],C[7],C[8],C[9],C[10],C[11],
                            C[12],C[13],C[14],C[15],C[16],C[17],C[18],C[19],C[20],C[21],C[22],C[23],
                            C[24]))
                                              
            #use the BFGS method to determine the point that minimizes the function
            tol = 1E-10
            if i == 0:
                initial_guess = [a0,c0]
            else:
                initial_guess = [latt_data[1][i-1],latt_data[3][i-1]]

            if withbounds == True:
                bounds    = [(xmin,xmax),(ymin,ymax)] #bounds of free energy grid
                result    = so.minimize(eighthorder_ploy,initial_guess,args=(C),
                                     method='L-BFGS-B',bounds = bounds,tol = tol)
            else:
                result    = so.minimize(eighthorder_ploy,initial_guess,args=(C),
                                     method='BFGS',tol = tol)

            #gather results from optimization routine
            latt_data[0][i] = Tmin+(Tmax-Tmin)/tempsteps*i
            latt_data[1][i] = result.x[0]
            latt_data[3][i] = result.x[1]
            
    elif num_unique == 3:
        for i in range(int(volnum)):
            x = np.append(x,latt_array[i][0]) # a 
            y = np.append(y,latt_array[i][1]) # b 
            t = np.append(t,latt_array[i][2]) # c
            
        #find max and min of the lattice parameters
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        tmin = min(t)
        tmax = max(t)
        
        #determine the density by looking at the required accuracy in the lattice parameters
        density_array = [(xmax-xmin)/desired_acc,(ymax-ymin)/desired_acc,(tmax-tmin)/desired_acc]
        density       = int(max(density_array))
            
        print('       number of new lattice points:         %s'%density) 
        print('       total number of extrapulation points: %s' %(density*density*density))       
        
        #generate list of coordinates for each temperature   
        for i in range(tempsteps):
            z = []
            for j in range(int(volnum)):
                z = np.append(z,(engy_array[j][0]+numatoms*free_array[j][i]))  #Add internal energy to each point.
                
            #internal test that they are all the same length
            try:
                x.shape[0] == y.shape[0] == z.shape[0] == t.shape[0]
            except:
                print('ERROR: The dimensions of either x,y, or z are unequal.')
                print('The dimensions of the arrays are %s %s %s.'%(x.shape[0],y.shape[0],z.shape[0]))
                sys.exit()

            #find best fit 6th order fit             
            A = np.c_[x**0*y**0*t**0+1, x**1*y**0*t**0, x**2*y**0*t**0, x**3*y**0*t**0, x**4*y**0*t**0,
                      x**0*y**1*t**0, x**1*y**1*t**0, x**2*y**1*t**0, x**3*y**1*t**0, x**4*y**1*t**0,
                      x**0*y**2*t**0, x**1*y**2*t**0, x**2*y**2*t**0, x**3*y**2*t**0, x**4*y**2*t**0,
                      x**0*y**3*t**0, x**1*y**3*t**0, x**2*y**3*t**0, x**3*y**3*t**0, x**4*y**3*t**0,
                      x**0*y**4*t**0, x**1*y**4*t**0, x**2*y**4*t**0, x**3*y**4*t**0, x**4*y**4*t**0,
                      x**0*y**0*t**1, x**1*y**0*t**1, x**2*y**0*t**1, x**3*y**0*t**1, x**4*y**0*t**1,
                      x**0*y**1*t**1, x**1*y**1*t**1, x**2*y**1*t**1, x**3*y**1*t**1, x**4*y**1*t**1,
                      x**0*y**2*t**1, x**1*y**2*t**1, x**2*y**2*t**1, x**3*y**2*t**1, x**4*y**2*t**1,
                      x**0*y**3*t**1, x**1*y**3*t**1, x**2*y**3*t**1, x**3*y**3*t**1, x**4*y**3*t**1,
                      x**0*y**4*t**1, x**1*y**4*t**1, x**2*y**4*t**1, x**3*y**4*t**1, x**4*y**4*t**1,
                      x**0*y**0*t**2, x**1*y**0*t**2, x**2*y**0*t**2, x**3*y**0*t**2, x**4*y**0*t**2,
                      x**0*y**1*t**2, x**1*y**1*t**2, x**2*y**1*t**2, x**3*y**1*t**2, x**4*y**1*t**2,
                      x**0*y**2*t**2, x**1*y**2*t**2, x**2*y**2*t**2, x**3*y**2*t**2, x**4*y**2*t**2,
                      x**0*y**3*t**2, x**1*y**3*t**2, x**2*y**3*t**2, x**3*y**3*t**2, x**4*y**3*t**2,
                      x**0*y**4*t**2, x**1*y**4*t**2, x**2*y**4*t**2, x**3*y**4*t**2, x**4*y**4*t**2,
                      x**0*y**0*t**3, x**1*y**0*t**3, x**2*y**0*t**3, x**3*y**0*t**3, x**4*y**0*t**3,
                      x**0*y**1*t**3, x**1*y**1*t**3, x**2*y**1*t**3, x**3*y**1*t**3, x**4*y**1*t**3,
                      x**0*y**2*t**3, x**1*y**2*t**3, x**2*y**2*t**3, x**3*y**2*t**3, x**4*y**2*t**3,
                      x**0*y**3*t**3, x**1*y**3*t**3, x**2*y**3*t**3, x**3*y**3*t**3, x**4*y**3*t**3,
                      x**0*y**4*t**3, x**1*y**4*t**3, x**2*y**4*t**3, x**3*y**4*t**3, x**4*y**4*t**3,
                      x**0*y**0*t**4, x**1*y**0*t**4, x**2*y**0*t**4, x**3*y**0*t**4, x**4*y**0*t**4,
                      x**0*y**1*t**4, x**1*y**1*t**4, x**2*y**1*t**4, x**3*y**1*t**4, x**4*y**1*t**4,
                      x**0*y**2*t**4, x**1*y**2*t**4, x**2*y**2*t**4, x**3*y**2*t**4, x**4*y**2*t**4,
                      x**0*y**3*t**4, x**1*y**3*t**4, x**2*y**3*t**4, x**3*y**3*t**4, x**4*y**3*t**4,
                      x**0*y**4*t**4, x**1*y**4*t**4, x**2*y**4*t**4, x**3*y**4*t**4, x**4*y**4*t**4]
            
            #C contains the coefficients to the polynomial we are trying to fit. 
            # if statement added to handle bad behavior in older versions of numpy
            try: 
                C,_,_,_ = np.linalg.lstsq(A, z,rcond=-1)
                 
            except FutureWarning:
                C,_,_,_ = np.linalg.lstsq(A, z,rcond=-1) 

            #print the coefficients, C, to a file as a function of temperature
            with open('out.coefficients','a') as f:
                f.write('%i %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s '\
                        '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s '\
                        '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s '\
                        '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s '\
                        '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n'
                       %(i,C[0],C[1],C[2],C[3],C[4],C[5],C[6],C[7],C[8],C[10],C[11],C[12],C[13],C[14],
                          C[15],C[16],C[17],C[18],C[19],C[20],C[21],C[22],C[23],C[24],C[25],C[26],C[27],C[28],C[29],
                          C[30],C[31],C[32],C[33],C[34],C[35],C[36],C[37],C[38],C[39],C[40],C[41],C[42],C[43],C[44],
                          C[45],C[46],C[47],C[48],C[49],C[50],C[51],C[51],C[52],C[53],C[54],C[55],C[56],C[57],C[58],
                          C[59],C[60],C[61],C[62],C[63],C[64],C[65],C[66],C[67],C[68],C[69],C[70],C[71],C[72],C[73],
                          C[74],C[75],C[76],C[77],C[78],C[79],C[80],C[81],C[82],C[83],C[84],C[85],C[86],C[87],C[88],
                          C[89],C[90],C[91],C[92],C[93],C[94],C[95],C[96],C[97],C[98],C[99],C[100],C[101],C[102],C[103],
                          C[104],C[105],C[106],C[107],C[108],C[109],C[110],C[111],C[112],C[113],C[114],C[115],C[116],C[117],C[118],
                          C[119],C[120],C[121],C[122],C[123],C[124]))
                
                                              
            #use the BFGS method to determine the point that minimizes the function
            tol = 1E-10
            if i == 0:
                initial_guess = [a0,b0,c0]
            else:
                initial_guess = [latt_data[1][i-1],latt_data[2][i-1],latt_data[3][i-1]]

            if withbounds == True:
                bounds    = [(xmin,xmax),(ymin,ymax),(tmin,tmax)] #bounds of free energy grid
                result    = so.minimize(sixthfourhorder_ploy,initial_guess,args=(C),
                                     method='L-BFGS-B',bounds = bounds,tol = tol)
            else:
                result    = so.minimize(sixthfourhorder_ploy,initial_guess,args=(C),
                                     method='BFGS',tol = tol)

            #gather results from optimization routine
            latt_data[0][i] = Tmin+(Tmax-Tmin)/tempsteps*i
            latt_data[1][i] = result.x[0]
            latt_data[2][i] = result.x[1]    
            latt_data[3][i] = result.x[2] 
    
    return latt_data,cell_data

def fit_EOS(volnum,num_unique,cell_data,engy_array,free_array,vol_array,withbounds):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@smn.uio.no
    
    Purpose: Fit equation of state using volume data
    
    Return: Arrays of bulk modulus data and cell data
    """
    #declare useful information
    a0              = cell_data[1]/ang_to_m
    b0              = cell_data[2]/ang_to_m
    c0              = cell_data[3]/ang_to_m
    numatoms        = float(cell_data[4]) 
    Tmin            = int(cell_data[23])
    Tmax            = int(cell_data[24])
    tempsteps       = int(cell_data[25])
    
    #declare arrays
    bulkT = np.empty(shape=(8,tempsteps))     
    
    print('   3 - Fitting the equations of state.' )
    print('       ---Brief pauses are normal---')
    if withbounds == True:
        print('       Bounds are being used.')
    
    plsq2 = []
    for i in range(tempsteps):
        z     = []
        for j in range(int(volnum)):
            z = np.append(z,(engy_array[j][0]+numatoms*free_array[j][i]))  #Add internal energy to each point.
        
        #internal test that they are all the same length
        try:
            vol_array.shape[0] == z.shape[0]
        except:
            print('ERROR: The dimensions of either volume or z are unequal.')
            print('The dimensions of the arrays are %s %s.'%(vol_array.shape[0],z.shape[0]))
            sys.exit()
        
        #define an initial guess for the parameters of the Birch_Murnaghan equation
        if z[0] > 0:
            print('ERROR: Check calculation of free energy or internal energy. Bad value encountered: %s' %z[0])
            sys.exit()
            E0 = 0 #reference energy (should be negative and in eV)
        else:
            E0 = z[0] #reference energy (should be negative and in eV)
        
        if i == 0:
            B0 = float(cell_data[27]/(10*160.21765))      #guess at bulk modulus 
            BP = 5.0       #guess at pressure dependence of bulk modulus
            V0 = a0*b0*c0  #guess for initial volume (in Å^3)
            x0 = np.array([E0,B0,BP,V0],dtype=float)
        else:
            x0 = plsq2
        
        if withbounds == True:
            bounds = ((-1000,0.0,-15,V0*0.6),(0,B0*1.2,15,V0*1.2))
        else:
            bounds = ((-np.inf,-np.inf,-np.inf,-np.inf),( np.inf,np.inf,np.inf,np.inf))
            
        #determine the least squared fit parameters
        try:
            plsq,covariance  = so.curve_fit(Birch_Murnaghan, vol_array, z, p0=x0, bounds = bounds, sigma=0.5*np.ones(shape=z.shape[0]))
            plsq2,covariance = so.curve_fit(Murnaghan,       vol_array, z, p0=x0, bounds = bounds, sigma=0.5*np.ones(shape=z.shape[0]))
        except RuntimeError:
            print('The E vs. V data is rather spread out.  Consider a better k-point grid')
            plsq,covariance  = so.curve_fit(Birch_Murnaghan, vol_array, z, p0=x0, bounds = bounds, sigma=3.0*np.ones(shape=z.shape[0]))
            plsq2,covariance = so.curve_fit(Murnaghan,       vol_array, z, p0=x0, bounds = bounds, sigma=3.0*np.ones(shape=z.shape[0]))
                  
        #store the bulk modulus as a function of temperature
        bulkT[0][i] = Tmin+(Tmax-Tmin)/tempsteps*i
        bulkT[1][i] = plsq[0]           #cohesive energy 
        bulkT[2][i] = plsq[1]*160.21765 #changes the unit to GPa (bulk modulus)
        bulkT[3][i] = plsq[2]           #unitless (pressure derivative)
        bulkT[4][i] = plsq2[0]          #cohesive energy
        bulkT[5][i] = plsq2[1]*160.21765 #changes the unit to GPa
        bulkT[6][i] = plsq2[2]           #unitless (pressure derivative)
                    
    return bulkT,cell_data

def find_CTE(volnum,num_unique,cell_data,latt_data,withpoly):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@smn.uio.no
    
    Purpose: Calculate the coefficients of thermal expansion
    
    Return:  
    """
    #declare useful information
    tempsteps  = int(cell_data[25])
    tmin       = int(cell_data[23])
    tmax       = int(cell_data[24])
    tempsteps  = int(cell_data[25])
    tspacing   = (tmax-tmin)/tempsteps
           
    #declare arrays
    xdata = latt_data[0][:] #temperature
    adata = latt_data[1][:] #a lattice
    bdata = latt_data[2][:] #b lattice
    cdata = latt_data[3][:] #c lattice 
    latt_der        = np.zeros(shape=(4,tempsteps)) 
    lattdata        = np.zeros(shape=(tempsteps,9))
    
    #poly fit ranges
    if withpoly[0] == True:
        fitlower   = float(withpoly[1])
        fitupper   = float(withpoly[2])
        print_poly = withpoly[3]
    else:
        fitlower = 0.1
        fitupper = 2
        
    #note that window must be an odd number!
    if int(tempsteps*0.01)%2 == 0: #number is odd
        window = int(tempsteps*0.01)+1
    else:
        window = int(tempsteps*0.01)
    
    with open('data_extraction','r') as datafile:
        for line in datafile:
            if 'Debye' in line:
                l = line.split()
                debye = l[1]
                
    #set lower and upper limit to fit
    fitrangelow  = int(fitlower*float(debye)/tspacing)
    fitrangeup   = int(fitupper*float(debye)/tspacing)
 
    print('   4 - Calculate the coefficients of thermal expansion.') 
    
    if num_unique == 1:
        if withpoly[0] == True:
            #fit the extracted data to a polynomial of order 8
            pa = np.polyfit(xdata[fitrangelow:fitrangeup],adata[fitrangelow:fitrangeup],8)#produces an array of coefficients, highest to lowest
                        
            #produces an equation using the previously calculated coefficients
            apoly = np.poly1d(pa)  
            
            #get array of values
            ahat = apoly(xdata)
            
            if print_poly:
                f1= open('out.poly_lattice','w')
                for i in range(len(ahat)):
                    f1.write('%s %s\n' %(xdata[i],ahat[i]))        
                f1.close()
                        
            #take the log of the lattice parameter
            logahat = np.zeros(shape=(len(ahat)))
            for i in range(len(adata)):
                logahat[i] = np.log(ahat[i])
                
            # calculate the derivative of the polynomial
            ahatder = np.gradient(logahat,tspacing) 
                                       
            for i in range(int(tempsteps)):
                latt_der[0][i] = xdata[i]
                latt_der[1][i] = ahatder[i]
                lattdata[i][0] = latt_der[1][i]
                lattdata[i][4] = latt_der[1][i]
                lattdata[i][8] = latt_der[1][i]
                
        elif withpoly[0] == False:
            #smooth data with running mean
            ahat = running_mean(adata,window) 
        
            #take the log of the lattice parameter
            for i in range(len(adata)):
                ahat[i] = np.log(ahat[i])
                
            # take derivative of ahat using a gradient        
            dera = np.gradient(ahat,tspacing)
            
            #filter to smooth
            for i in range(10):
                dera = running_mean(dera,window)
                                
            for i in range(int(tempsteps)):
                latt_der[0][i] = xdata[i]
                latt_der[1][i] = dera[i]
                lattdata[i][0] = latt_der[1][i]
                lattdata[i][4] = latt_der[1][i]
                lattdata[i][8] = latt_der[1][i]
        else:
            print('Something bad happened in find_CTE. Contact developer.')
            sys.exit()
                    
    elif num_unique == 2:
        if withpoly[0] == True:
            #fit the extracted data to a polynomial of order 6
            pa = np.polyfit(xdata[fitrangelow:fitrangeup],adata[fitrangelow:fitrangeup],8)  #produces an array of coefficients, highest to lowest
            pc = np.polyfit(xdata[fitrangelow:fitrangeup],cdata[fitrangelow:fitrangeup],8)
            
            #produces an equation using the previously calculated coefficients
            apoly = np.poly1d(pa)  
            cpoly = np.poly1d(pc)
            
            #get array of values
            ahat = apoly(xdata)
            chat = cpoly(xdata)
            
            if print_poly:
                f1= open('out.poly_lattice','w')
                for i in range(len(ahat)):
                    f1.write('%s %s %s\n' %(xdata[i],ahat[i],chat[i]))
                f1.close()
            
            #take the log of the lattice parameter
            logahat = np.zeros(shape=(len(ahat)))
            logchat = np.zeros(shape=(len(chat)))
            for i in range(len(adata)):
                logahat[i] = np.log(ahat[i])
                logchat[i] = np.log(chat[i])
                
            # calculate the derivative of the polynomial
            ahatder = np.gradient(logahat,tspacing) 
            chatder = np.gradient(logchat,tspacing) 
                                       
            for i in range(int(tempsteps)):
                latt_der[0][i] = xdata[i]
                latt_der[1][i] = ahatder[i]
                latt_der[2][i] = chatder[i]
                lattdata[i][0] = latt_der[1][i]
                lattdata[i][4] = latt_der[1][i]
                lattdata[i][8] = latt_der[2][i]
                
        elif withpoly[0] == False:
            #smooth data with running mean
            ahat = running_mean(adata,window)
            chat = running_mean(cdata,window)
        
            #take the log of the lattice parameter
            for i in range(len(adata)):
                ahat[i] = np.log(ahat[i])
                chat[i] = np.log(chat[i])
                
            # take derivative of ahat using a gradient        
            dera = np.gradient(ahat,tspacing)
            derc = np.gradient(chat,tspacing)
            
            #filter to smooth
            for i in range(10):
                dera = running_mean(dera,window)
                derc = running_mean(derc,window)
                                
            for i in range(int(tempsteps)):
                latt_der[0][i] = xdata[i]
                latt_der[1][i] = dera[i]
                latt_der[2][i] = derc[i]
                lattdata[i][0] = latt_der[1][i]
                lattdata[i][4] = latt_der[1][i]
                lattdata[i][8] = latt_der[2][i]
        else:
            print('Something bad happened in find_CTE. Contact developer.')
            sys.exit()
                   
    elif num_unique == 3:
        if withpoly[0] == True:
            #fit the extracted data to a polynomial of order 6
            pa = np.polyfit(xdata[fitrangelow:fitrangeup],adata[fitrangelow:fitrangeup],8)  #produces an array of coefficients, highest to lowest
            pb = np.polyfit(xdata[fitrangelow:fitrangeup],bdata[fitrangelow:fitrangeup],8)
            pc = np.polyfit(xdata[fitrangelow:fitrangeup],cdata[fitrangelow:fitrangeup],8)
            
            #produces an equation using the previously calculated coefficients
            apoly = np.poly1d(pa) 
            bpoly = np.poly1d(pb)
            cpoly = np.poly1d(pc)
            
            #get array of values
            ahat = apoly(xdata)
            bhat = bpoly(xdata)
            chat = cpoly(xdata)
            
            if print_poly:
                f1= open('out.poly_lattice','w')
                for i in range(len(ahat)):
                    f1.write('%s %s %s %s\n' %(xdata[i],ahat[i],bhat[i],chat[i]))
                f1.close()
            
            #take the log of the lattice parameter
            logahat = np.zeros(shape=(len(ahat)))
            logbhat = np.zeros(shape=(len(bhat)))
            logchat = np.zeros(shape=(len(chat)))
            for i in range(len(adata)):
                logahat[i] = np.log(ahat[i])
                logbhat[i] = np.log(bhat[i])
                logchat[i] = np.log(chat[i])
                
            # calculate the derivative of the polynomial
            ahatder = np.gradient(logahat,tspacing) 
            bhatder = np.gradient(logbhat,tspacing)
            chatder = np.gradient(logchat,tspacing) 
                                       
            for i in range(int(tempsteps)):
                latt_der[0][i] = xdata[i]
                latt_der[1][i] = ahatder[i]
                latt_der[2][i] = bhatder[i]
                latt_der[3][i] = chatder[i]
                lattdata[i][0] = latt_der[1][i]
                lattdata[i][4] = latt_der[1][i]
                lattdata[i][8] = latt_der[1][i]
                
        elif withpoly[0] == False:
            #smooth data with running mean
            ahat = running_mean(adata,window)
            bhat = running_mean(bdata,window)
            chat = running_mean(cdata,window)
        
            #take the log of the lattice parameter
            for i in range(len(adata)):
                ahat[i] = np.log(ahat[i])
                bhat[i] = np.log(bhat[i])
                chat[i] = np.log(chat[i])
                
            # take derivative of ahat using a gradient        
            dera = np.gradient(ahat,tspacing)
            derb = np.gradient(bhat,tspacing)
            derc = np.gradient(chat,tspacing)
            
            #filter to smooth
            for i in range(10):
                dera = running_mean(dera,window)
                derb = running_mean(derb,window)
                derc = running_mean(derc,window)
                                
            for i in range(int(tempsteps)):
                latt_der[0][i] = xdata[i]
                latt_der[1][i] = dera[i]
                latt_der[2][i] = derb[i]
                latt_der[3][i] = derc[i]
                lattdata[i][0] = latt_der[1][i]
                lattdata[i][4] = latt_der[2][i]
                lattdata[i][8] = latt_der[3][i]
        else:
            print('Something bad happened in find_CTE. Contact developer.')
            sys.exit()        
            
    #save lattice expansion coefficients to cell_data for use by the rest of the program
    cell_data[10] = lattdata  
        
    return lattdata,cell_data

def find_CP(volnum,num_unique,cell_data,lattder,bulkT,total_mass):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Calculate the specific heat at constant pressure
    
    Return: Array of specific heat at constant pressure, constant volume, and cell data
    
    """
    #declare useful information 
    vol             = cell_data[0]      # DFT volume of unit cell at T=0K in m**3 
    a0              = cell_data[1]/ang_to_m
    b0              = cell_data[2]/ang_to_m
    c0              = cell_data[3]/ang_to_m
    files           = cell_data[19]
    tempsteps       = int(cell_data[25])
    foundfile       = False
        
    #declare arrays
    sheat           = np.zeros(shape=(4,tempsteps))

    print('   5 - Calculate Cp.')
    if num_unique == 1:
        for file in files:
            #use the file name to determine the lattice constants
            l = file.split('_')
            a = float(l[3]) # a lattice
            if np.abs(a-a0) < difftol:
                foundfile = True
                cvfile = file
                
        if not foundfile:
            print('Free energy file corresponding to the relaxed DFT lattice parameters not found!')
            sys.exit()
            
        i = 0
        with open('free_energies/'+cvfile,'r') as f:
            for num,line in enumerate(f,0):
                l = line.strip('\n').split()
                #this file contains 0- temperature, 1- vibrational free energy, 2- entropy, 3- specific heat
                sheat[0][i] = float(l[0])
                sheat[1][i] = float(l[3])
                i += 1
        
        #determine cp using thermodynamics
        for i in range(sheat.shape[1]):
            alphaavg    = (lattder[i][0] + lattder[i][4]+ lattder[i][8])/3.0
            sheat[2][i] = sheat[1][i] + vol*sheat[0][i]*alphaavg**2*bulkT[2][i]*J_to_cal/total_mass
                        
    elif num_unique == 2:
        for file in files:
            #use the file name to determine the lattice constants
            l = file.split('_')
            a = float(l[3]) # a lattice
            c = float(l[5]) # c latice 
            if np.abs(a-a0) < difftol and np.abs(c-c0) < difftol:
                foundfile = True
                cvfile = file
                
        if not foundfile:
            print('Free energy file corresponding to the relaxed DFT lattice parameters not found!')
            sys.exit()
            
        i = 0
        with open('free_energies/'+cvfile,'r') as f:
            for num,line in enumerate(f,0):
                l = line.strip('\n').split()
                #this file contains 0- temperature, 1- vibrational free energy, 2- entropy, 3- specific heat
                sheat[0][i] = float(l[0])
                sheat[1][i] = float(l[3])
                i += 1
        
        #determine cp using thermodynamics
        for i in range(sheat.shape[1]):
            alphaavg    = (lattder[i][0] + lattder[i][4]+ lattder[i][8])/3.0
            sheat[2][i] = sheat[1][i] + vol*sheat[0][i]*alphaavg**2*bulkT[2][i]*J_to_cal/total_mass
                
    elif num_unique == 3:
        for file in files:
            #use the file name to determine the lattice constants
            l = file.split('_')
            a = float(l[3]) # a lattice
            b = float(l[4]) # b lattice
            c = float(l[5]) # c latice 
            if np.abs(a-a0) < difftol and np.abs(c-c0) < difftol and np.abs(b-b0) < difftol:
                foundfile = True
                cvfile = file
                
        if not foundfile:
            print('Free energy file corresponding to the relaxed DFT lattice parameters not found!')
            sys.exit()
            
        i = 0
        with open('free_energies/'+cvfile,'r') as f:
            for num,line in enumerate(f,0):
                l = line.strip('\n').split()
                #this file contains 0- temperature, 1- vibrational free energy, 2- entropy, 3- specific heat
                sheat[0][i] = float(l[0])
                sheat[1][i] = float(l[3])
                i += 1
        
        #determine cp using thermodynamics
        for i in range(sheat.shape[1]):
            alphaavg    = (lattder[i][0] + lattder[i][4]+ lattder[i][8])/3.0
            sheat[2][i] = sheat[1][i] + vol*sheat[0][i]*alphaavg**2*bulkT[2][i]*J_to_cal/total_mass
                        
    return sheat,cell_data

def print_all(volnum,num_unique,cell_data,free_array,latt_data,lattder,bulkT,sheat):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Print all calculated data to the execution folder
    
    Return: None
    """
    #declare useful information
    Tmin            = int(cell_data[23])
    Tmax            = int(cell_data[24])
    tempsteps       = int(cell_data[25])
    
    print('   6 - Printing output files. ') 
        
    #print free energy to a file
    freefilename = 'out.free_energy_vs_temp'
    f1= open(freefilename,'w')
    f1.write('# Free energy vs Temp\n #temp\tF1\tF2\t...\n')
    temp = 0
    for i in range(int(cell_data[25])):
        temp = Tmin+(Tmax-Tmin)/tempsteps*i
        for j in range(int(volnum)):
            if j == int(volnum-1):
                f1.write('%s\n' %free_array[j][i])
            elif j == 0:
                f1.write('%s %s ' %(temp,free_array[j][i]))
            else:
                f1.write('%s '%free_array[j][i])
    f1.close()
    
    #generate gnuplot file 
    gen_GNUPLOT(freefilename,'free_energy_total.gnuplot',int(volnum),'free_energy') 
    
    #print this data to a file
    f1= open('out.expansion_coeffs','w')
    f1.write('# expansion coefficients \n#temp \t a \n' )
    for i in range(latt_data.shape[1]):
        f1.write('%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s\n'
                 %(latt_data[0][i],
                   lattder[i][0],lattder[i][1],lattder[i][2],
                   lattder[i][3],lattder[i][4],lattder[i][5],
                   lattder[i][6],lattder[i][7],lattder[i][8])) 
    f1.close()
        
    if num_unique == 1:
        #now that the minimum coordinates have been found we can print them to a file
        f1= open('out.thermal_expansion','w')
        f1.write('# Thermal lattice parameters \n#temp \t a\n' )
        for i in range(latt_data.shape[1]):
            f1.write('%s \t%s\n' %(latt_data[0][i],latt_data[1][i]))
        f1.close()
        
        #generate gnuplot file 
        gen_GNUPLOT('out.thermal_expansion','thermal_expansion.gnuplot',2,'thermal_lattice')
               
        #generate gnuplot file 
        gen_GNUPLOT('out.expansion_coeffs','expansion_coeffs.gnuplot',2,'expansion')
        
    elif num_unique == 2:
        #now that the minimum coordinates have been found we can print them to a file
        f1= open('out.thermal_expansion','w')
        f1.write('# Thermal lattice parameters \n#temp \t a \t c\n' )
        for i in range(latt_data.shape[1]):
            f1.write('%s \t%s \t%s\n' %(latt_data[0][i],latt_data[1][i],latt_data[3][i]))
        f1.close()
        
        #generate gnuplot file 
        gen_GNUPLOT('out.thermal_expansion','thermal_expansion.gnuplot',3,'thermal_lattice')
                
        #generate gnuplot file 
        gen_GNUPLOT('out.expansion_coeffs','expansion_coeffs.gnuplot',3,'expansion')
        
    elif num_unique == 3:
        #now that the minimum coordinates have been found we can print them to a file
        f1= open('out.thermal_expansion','w')
        f1.write('# Thermal lattice parameters \n#temp \t a \t b \t c\n' )
        for i in range(latt_data.shape[1]):
            f1.write('%s \t%s \t%s \t%s\n' %(latt_data[0][i],latt_data[1][i],latt_data[2][i],latt_data[3][i]))
        f1.close()
        
        #generate gnuplot file 
        gen_GNUPLOT('out.thermal_expansion','thermal_expansion.gnuplot',4,'thermal_lattice')
                
        #generate gnuplot file 
        gen_GNUPLOT('out.expansion_coeffs','expansion_coeffs.gnuplot',3,'expansion')
        
    #print this data to a file
    f1= open('out.isothermal_bulk','w')
    f1.write('# isothermal bulk modulus \n#temp \t E0 \t B \t dB/dp etc. \n' )
    for i in range(bulkT.shape[1]):
        f1.write('%s \t%s \t%s \t%s \t%s \t%s \t%s\n'
                 %(bulkT[0][i],bulkT[1][i],bulkT[2][i],bulkT[3][i],bulkT[4][i],bulkT[5][i],bulkT[6][i])) 
    f1.close()
    
    #generate gnuplot file 
    gen_GNUPLOT('out.isothermal_bulk','isothermal_bulk.gnuplot',2,'bulk_modulus')
        
    #print this data to a file
    f1= open('out.cv_cp','w')
    f1.write('# specific heat \n#temp \t cv \t cp\n' )
    for i in range(latt_data.shape[1]):
        f1.write('%s \t%s \t%s\n' %(sheat[0][i],sheat[1][i],sheat[2][i])) 
    f1.close()
    #generate gnuplot file 
    gen_GNUPLOT('out.cv_cp','specific_heat.gnuplot',3,'cv_cp')
    
    return None

def Birch_Murnaghan(vol,E0, B0, BP,V0):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Define the Birch_Murnaghan equation of state that relates energy vs volume
             Note that this equation comes from Reference Phys. Rev. B 70, 224107 (2004)
    
    Returns: Energy 
    """    
    #define equation 
    eta = (vol/V0)**(1.0/3.0)
    E = E0 + 9.0*B0*V0/16.0 * (eta**2-1.0)**2 * (6.0 + BP*(eta**2-1.0) - 4.0*eta**2)
    
    return E

def Murnaghan(vol,E0, B0, BP,V0):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Define the Murnaghan equation of state from Phys. Rev. B 28, 5480 (1983)
    
    Return: Energy
    """
    E = E0 + B0/BP * vol * ((V0/vol)**BP/(BP-1.0)+1.0) - V0*B0/(BP-1.0)
    
    return E

def fourthorder_ploy(x, cof):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determine the 6th order polynomial of the free energy for a fixed
             temperature. Note that coord is an array passed from the optimizer
    
    Output:  Returns function which is used by the optimizer to find the global minimum
             
    """    
    # cof contains the fit coefficients of the 6th order polynomial of a and c
    func = (cof[0]*x[0]**0 + cof[1]*x[0]**1 + cof[2]*x[0]**2 + cof[3]*x[0]**3 + cof[4]*x[0]**4)
    
    return func

def eighthorder_ploy(x, cof):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determine the 12th order polynomial of the free energy for a fixed
             temperature. Note that coord is an array passed from the optimizer
    
    Output:  Returns function which is used by the optimizer to find the global minimum
             
    """    
    # cof contains the fit coefficients of the 6th order polynomial of a and c
    func = (cof[ 0]*x[0]**0*x[1]**0 + cof[ 1]*x[0]**1*x[1]**0 + cof[ 2]*x[0]**2*x[1]**0 + cof[ 3]*x[0]**3*x[1]**0 + cof[ 4]*x[0]**4*x[1]**0 +
            cof[ 5]*x[0]**0*x[1]**1 + cof[ 6]*x[0]**1*x[1]**1 + cof[ 7]*x[0]**2*x[1]**1 + cof[ 8]*x[0]**3*x[1]**1 + cof[ 9]*x[0]**4*x[1]**1 + 
            cof[10]*x[0]**0*x[1]**2 + cof[11]*x[0]**1*x[1]**2 + cof[12]*x[0]**2*x[1]**2 + cof[13]*x[0]**3*x[1]**2 + cof[14]*x[0]**4*x[1]**2 + 
            cof[15]*x[0]**0*x[1]**3 + cof[16]*x[0]**1*x[1]**3 + cof[17]*x[0]**2*x[1]**3 + cof[18]*x[0]**3*x[1]**3 + cof[19]*x[0]**4*x[1]**3 + 
            cof[20]*x[0]**0*x[1]**4 + cof[21]*x[0]**1*x[1]**4 + cof[22]*x[0]**2*x[1]**4 + cof[23]*x[0]**3*x[1]**4 + cof[24]*x[0]**4*x[1]**4 )
    
    return func

def sixthfourhorder_ploy(x, cof):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determine the 12th order polynomial of the free energy for a fixed
             temperature. Note that coord is an array passed from the optimizer
    
    Output:  Returns function which is used by the optimizer to find the global minimum
             
    """    
    # cof contains the fit coefficients of the 6th order polynomial of a and c
    func = (cof[ 0]*x[0]**0*x[1]**0*x[2]**0 + cof[ 1]*x[0]**1*x[1]**0*x[2]**0 + cof[ 2]*x[0]**2*x[1]**0*x[2]**0 + cof[ 3]*x[0]**3*x[1]**0*x[2]**0 + cof[ 4]*x[0]**4*x[1]**0*x[2]**0 +
            cof[ 5]*x[0]**0*x[1]**1*x[2]**0 + cof[ 6]*x[0]**1*x[1]**1*x[2]**0 + cof[ 7]*x[0]**2*x[1]**1*x[2]**0 + cof[ 8]*x[0]**3*x[1]**1*x[2]**0 + cof[ 9]*x[0]**4*x[1]**1*x[2]**0 + 
            cof[10]*x[0]**0*x[1]**2*x[2]**0 + cof[11]*x[0]**1*x[1]**2*x[2]**0 + cof[12]*x[0]**2*x[1]**2*x[2]**0 + cof[13]*x[0]**3*x[1]**2*x[2]**0 + cof[14]*x[0]**4*x[1]**2*x[2]**0 + 
            cof[15]*x[0]**0*x[1]**3*x[2]**0 + cof[16]*x[0]**1*x[1]**3*x[2]**0 + cof[17]*x[0]**2*x[1]**3*x[2]**0 + cof[18]*x[0]**3*x[1]**3*x[2]**0 + cof[19]*x[0]**4*x[1]**3*x[2]**0 + 
            cof[20]*x[0]**0*x[1]**4*x[2]**0 + cof[21]*x[0]**1*x[1]**4*x[2]**0 + cof[22]*x[0]**2*x[1]**4*x[2]**0 + cof[23]*x[0]**3*x[1]**4*x[2]**0 + cof[24]*x[0]**4*x[1]**4*x[2]**0 + 
    
            cof[25]*x[0]**0*x[1]**0*x[2]**1 + cof[26]*x[0]**1*x[1]**0*x[2]**1 + cof[27]*x[0]**2*x[1]**0*x[2]**1 + cof[28]*x[0]**3*x[1]**0*x[2]**1 + cof[29]*x[0]**4*x[1]**0*x[2]**1 +
            cof[30]*x[0]**0*x[1]**1*x[2]**1 + cof[31]*x[0]**1*x[1]**1*x[2]**1 + cof[32]*x[0]**2*x[1]**1*x[2]**1 + cof[33]*x[0]**3*x[1]**1*x[2]**1 + cof[34]*x[0]**4*x[1]**1*x[2]**1 + 
            cof[35]*x[0]**0*x[1]**2*x[2]**1 + cof[36]*x[0]**1*x[1]**2*x[2]**1 + cof[37]*x[0]**2*x[1]**2*x[2]**1 + cof[38]*x[0]**3*x[1]**2*x[2]**1 + cof[39]*x[0]**4*x[1]**2*x[2]**1 + 
            cof[40]*x[0]**0*x[1]**3*x[2]**1 + cof[41]*x[0]**1*x[1]**3*x[2]**1 + cof[42]*x[0]**2*x[1]**3*x[2]**1 + cof[43]*x[0]**3*x[1]**3*x[2]**1 + cof[44]*x[0]**4*x[1]**3*x[2]**1 + 
            cof[45]*x[0]**0*x[1]**4*x[2]**1 + cof[46]*x[0]**1*x[1]**4*x[2]**1 + cof[47]*x[0]**2*x[1]**4*x[2]**1 + cof[48]*x[0]**3*x[1]**4*x[2]**1 + cof[49]*x[0]**4*x[1]**4*x[2]**1 +
            
            cof[50]*x[0]**0*x[1]**0*x[2]**2 + cof[51]*x[0]**1*x[1]**0*x[2]**2 + cof[52]*x[0]**2*x[1]**0*x[2]**2 + cof[53]*x[0]**3*x[1]**0*x[2]**2 + cof[54]*x[0]**4*x[1]**0*x[2]**2 +
            cof[55]*x[0]**0*x[1]**1*x[2]**2 + cof[56]*x[0]**1*x[1]**1*x[2]**2 + cof[57]*x[0]**2*x[1]**1*x[2]**2 + cof[58]*x[0]**3*x[1]**1*x[2]**2 + cof[59]*x[0]**4*x[1]**1*x[2]**2 + 
            cof[60]*x[0]**0*x[1]**2*x[2]**2 + cof[61]*x[0]**1*x[1]**2*x[2]**2 + cof[62]*x[0]**2*x[1]**2*x[2]**2 + cof[63]*x[0]**3*x[1]**2*x[2]**2 + cof[64]*x[0]**4*x[1]**2*x[2]**2 + 
            cof[65]*x[0]**0*x[1]**3*x[2]**2 + cof[66]*x[0]**1*x[1]**3*x[2]**2 + cof[67]*x[0]**2*x[1]**3*x[2]**2 + cof[68]*x[0]**3*x[1]**3*x[2]**2 + cof[69]*x[0]**4*x[1]**3*x[2]**2 + 
            cof[70]*x[0]**0*x[1]**4*x[2]**2 + cof[71]*x[0]**1*x[1]**4*x[2]**2 + cof[72]*x[0]**2*x[1]**4*x[2]**2 + cof[73]*x[0]**3*x[1]**4*x[2]**2 + cof[74]*x[0]**4*x[1]**4*x[2]**2 +
            
            cof[75]*x[0]**0*x[1]**0*x[2]**3 + cof[76]*x[0]**1*x[1]**0*x[2]**3 + cof[77]*x[0]**2*x[1]**0*x[2]**3 + cof[78]*x[0]**3*x[1]**0*x[2]**3 + cof[79]*x[0]**4*x[1]**0*x[2]**3 +
            cof[80]*x[0]**0*x[1]**1*x[2]**3 + cof[81]*x[0]**1*x[1]**1*x[2]**3 + cof[82]*x[0]**2*x[1]**1*x[2]**3 + cof[83]*x[0]**3*x[1]**1*x[2]**3 + cof[84]*x[0]**4*x[1]**1*x[2]**3 + 
            cof[85]*x[0]**0*x[1]**2*x[2]**3 + cof[86]*x[0]**1*x[1]**2*x[2]**3 + cof[87]*x[0]**2*x[1]**2*x[2]**3 + cof[88]*x[0]**3*x[1]**2*x[2]**3 + cof[89]*x[0]**4*x[1]**2*x[2]**3 + 
            cof[90]*x[0]**0*x[1]**3*x[2]**3 + cof[91]*x[0]**1*x[1]**3*x[2]**3 + cof[92]*x[0]**2*x[1]**3*x[2]**3 + cof[93]*x[0]**3*x[1]**3*x[2]**3 + cof[94]*x[0]**4*x[1]**3*x[2]**3 + 
            cof[95]*x[0]**0*x[1]**4*x[2]**3 + cof[96]*x[0]**1*x[1]**4*x[2]**3 + cof[97]*x[0]**2*x[1]**4*x[2]**3 + cof[98]*x[0]**3*x[1]**4*x[2]**3 + cof[99]*x[0]**4*x[1]**4*x[2]**3 +
            
            cof[100]*x[0]**0*x[1]**0*x[2]**4 + cof[101]*x[0]**1*x[1]**0*x[2]**4 + cof[102]*x[0]**2*x[1]**0*x[2]**4 + cof[103]*x[0]**3*x[1]**0*x[2]**4 + cof[104]*x[0]**4*x[1]**0*x[2]**4 +
            cof[105]*x[0]**0*x[1]**1*x[2]**4 + cof[106]*x[0]**1*x[1]**1*x[2]**4 + cof[107]*x[0]**2*x[1]**1*x[2]**4 + cof[108]*x[0]**3*x[1]**1*x[2]**4 + cof[109]*x[0]**4*x[1]**1*x[2]**4 + 
            cof[110]*x[0]**0*x[1]**2*x[2]**4 + cof[111]*x[0]**1*x[1]**2*x[2]**4 + cof[112]*x[0]**2*x[1]**2*x[2]**4 + cof[113]*x[0]**3*x[1]**2*x[2]**4 + cof[114]*x[0]**4*x[1]**2*x[2]**4 + 
            cof[115]*x[0]**0*x[1]**3*x[2]**4 + cof[116]*x[0]**1*x[1]**3*x[2]**4 + cof[117]*x[0]**2*x[1]**3*x[2]**4 + cof[118]*x[0]**3*x[1]**3*x[2]**4 + cof[119]*x[0]**4*x[1]**3*x[2]**4 + 
            cof[120]*x[0]**0*x[1]**4*x[2]**4 + cof[121]*x[0]**1*x[1]**4*x[2]**4 + cof[122]*x[0]**2*x[1]**4*x[2]**4 + cof[123]*x[0]**3*x[1]**4*x[2]**4 + cof[124]*x[0]**4*x[1]**4*x[2]**4 )
    
    
    return func

def running_mean(x, N):
    """
    From stackoverflow...  Why is this not a numpy function?
    
    """
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N%2 == 0:
            a, b = i - (N-1)//2, i + (N-1)//2 + 2
        else:
            a, b = i - (N-1)//2, i + (N-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
        
    return out

def gen_GNUPLOT(plotfile,filename,numtoplot,plottype):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@smn.uio.no
    
    Purpose: Generate a gnuplot file to plot the plotfile using the file filename
    
    Return: None
    """
    outputline = 'set output "'+filename+'.eps"\n'
    if plottype == 'free_energy':
        f1= open(filename,'w')
        f1.write('# Automatically generated gnuplot file\n' )
        f1.write('# File created by Nicholas Pike using ACTE.py\n')
        f1.write('######################################################\n\n')
        f1.write('reset\n')
        f1.write('set terminal postscript eps enhanced color font "Helvetica,18" lw 1\n')
        f1.write(outputline)
        f1.write('\n#set line styles\n')
        f1.write('set style line 1 lc rgb "blue"\n')
        f1.write('set style line 2 lc rgb "black"\n')
        f1.write('set style line 3 lc rgb "red"\n')
        f1.write('#set yrange [] #automatically generated\n')
        f1.write('#set xrange [] #automatically generated\n')
        f1.write('\n#Plot data\n')
        for i in range(int(numtoplot)-1):
            if i == 0:
                f1.write('plot "%s" u 1:%i with points ls 1 notitle, ' %(plotfile,i+2))
            elif i == numtoplot-2:
                f1.write('"%s" u 1:%i with points ls 1 notitle ' %(plotfile,i+2))
            else:
                f1.write('"%s" u 1:%i with points ls 1 notitle,  ' %(plotfile,i+2))
        
        f1.write('\nset output\n\n')
        f1.write('# End gnuplot file for plotting')
        f1.close()
        
    elif plottype == 'thermal_lattice':
        f1= open(filename,'w')
        f1.write('# Automatically generated gnuplot file\n' )
        f1.write('# File created by Nicholas Pike using ACTE.py\n')
        f1.write('######################################################\n\n')
        f1.write('reset\n')
        f1.write('set terminal postscript eps enhanced color font "Helvetica,18" lw 1\n')
        f1.write(outputline)
        f1.write('\n#set line styles\n')
        f1.write('set style line 1 lc rgb "blue"\n')
        f1.write('set style line 2 lc rgb "black"\n')
        f1.write('set style line 3 lc rgb "red"\n')
        f1.write('set xrange ['+tmin+':'+tmax+'] #automatically generated\n')
        f1.write('#set yrange [] #automatically generated\n')
        f1.write('set ylabel "a-a_{exp} ({\305})"\n')
        f1.write('set xlabel "Temperature (K)"  \n')   
        f1.write('shift=0  #shift to be changed by user')
        f1.write('\n#Plot data\n')
        for i in range(int(numtoplot)-1):
            if i == 0:
                f1.write('plot "%s" u 1:($%i-shift) with points ls 1 notitle, ' %(plotfile,i+2))
            elif i == numtoplot-2:
                f1.write('"%s" u 1:($%i-shift) with points ls 1 notitle ' %(plotfile,i+2))
            else:
                f1.write('"%s" u 1:($%i-shift) with points ls 1 notitle,  ' %(plotfile,i+2))
        
        f1.write('\nset output\n\n')
        f1.write('# End gnuplot file for plotting')
        f1.close()
   
    elif plottype == 'expansion':
        f1= open(filename,'w')
        f1.write('# Automatically generated gnuplot file\n' )
        f1.write('# File created by Nicholas Pike using ACTE.py\n')
        f1.write('######################################################\n\n')
        f1.write('reset\n')
        f1.write('set terminal postscript eps enhanced color font "Helvetica,18" lw 1\n')
        f1.write(outputline)
        f1.write('\n#set line styles\n')
        f1.write('set style line 1 lc rgb "blue"\n')
        f1.write('set style line 2 lc rgb "black"\n')
        f1.write('set style line 3 lc rgb "red"\n')
        f1.write('set xrange ['+tmin+':'+tmax+'] #automatically generated\n')
        f1.write('#set yrange [] #automatically generated\n')
        f1.write('set ylabel "{/Symbol a}_a (10^{-6}/K)"\n')
        f1.write('set xlabel "Temperature (K)"  \n')   
        f1.write('\n#Plot data\n')
        for i in range(int(numtoplot)-1):
            if i == 0:
                f1.write('plot "%s" u 1:%i with points ls 1 notitle, ' %(plotfile,i+2))
            elif i == numtoplot-2:
                f1.write('"%s" u 1:%i with points ls 1 notitle ' %(plotfile,i+2))
            else:
                f1.write('"%s" u 1:%i with points ls 1 notitle,  ' %(plotfile,i+2))
        
        f1.write('\nset output\n\n')
        f1.write('# End gnuplot file for plotting')
        f1.close()
        
    elif plottype == 'bulk_modulus':
        f1= open(filename,'w')
        f1.write('# Automatically generated gnuplot file\n' )
        f1.write('# File created by Nicholas Pike using ACTE.py\n')
        f1.write('######################################################\n\n')
        f1.write('reset\n')
        f1.write('set terminal postscript eps enhanced color font "Helvetica,18" lw 1\n')
        f1.write(outputline)
        f1.write('\n#set line styles\n')
        f1.write('set style line 1 lc rgb "blue"\n')
        f1.write('set style line 2 lc rgb "black"\n')
        f1.write('set style line 3 lc rgb "red"\n')
        f1.write('set xrange ['+tmin+':'+tmax+'] #automatically generated\n')
        f1.write('#set yrange [] #automatically generated\n')
        f1.write('set y2range[-5:10]\n')
        f1.write('set ylabel "Bulk Modulus (GPa)"\n')
        f1.write('set y2label "dB/dp ()"\n')
        f1.write('set xlabel "Temperature (K)"  \n')   
        f1.write('\n#Plot data\n')
        for i in range(int(numtoplot)-1):
            if i == 0:
                f1.write('plot "%s" u 1:%i with points ls 1 notitle, ' %(plotfile,i+2))
            elif i == numtoplot-2:
                f1.write('"%s" u 1:%i with points ls 1 notitle ' %(plotfile,i+2))
            else:
                f1.write('"%s" u 1:%i with points ls 1 notitle,  ' %(plotfile,i+2))
        
        f1.write('\nset output\n\n')
        f1.write('# End gnuplot file for plotting')
        f1.close()    
        
    elif plottype == 'cv_cp':
        f1= open(filename,'w')
        f1.write('# Automatically generated gnuplot file\n' )
        f1.write('# File created by Nicholas Pike using ACTE.py\n')
        f1.write('######################################################\n\n')
        f1.write('reset\n')
        f1.write('set terminal postscript eps enhanced color font "Helvetica,18" lw 1\n')
        f1.write(outputline)
        f1.write('\n#set line styles\n')
        f1.write('set style line 1 lc rgb "blue"\n')
        f1.write('set style line 2 lc rgb "black"\n')
        f1.write('set style line 3 lc rgb "red"\n')
        f1.write('set xrange ['+tmin+':'+tmax+'] #automatically generated\n')
        f1.write('#set yrange [] #automatically generated\n')
        f1.write('#set y2range[]\n')
        f1.write('set ylabel "C (cal/g K)"\n')
        f1.write('set y2label "Log(Cp-Cv)"\n')
        f1.write('set xlabel "Temperature (K)"  \n')   
        f1.write('\n#Plot data\n')
        for i in range(int(numtoplot)-1):
            if i == 0:
                f1.write('plot "%s" u 1:%i with points ls 1 notitle, ' %(plotfile,i+2))
            elif i == numtoplot-2:
                f1.write('"%s" u 1:%i with points ls 1 notitle ' %(plotfile,i+2))
            else:
                f1.write('"%s" u 1:%i with points ls 1 notitle,  ' %(plotfile,i+2))
        
        f1.write('\nset output\n\n')
        f1.write('# End gnuplot file for plotting')
        f1.close() 
        
    return None

def gather_outcar():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Gather information from outcar files to be used in calculations of
             the thermal expansion coefficients.  This function will first check '
             for the existance of the file, and if not found, generate it.  If 
             the file is found, then it will check the values of the properties 
             currently in the file to see if they match with the current 
             calculation.  If they do not match, error messages are printed.
             
    Return: None
    """
    #properties to get
    printfilename = '../data_extraction'
    alat = 0
    blat = 0
    clat = 0
    vol  = 0
    natom = 0
    dietensor  = np.zeros(shape=(3,3))
    elatensor  = np.zeros(shape=(6,6))
    bectensor  = []
    infilename = 'OUTCAR'
    
    #check for existance of printfile and its contents
    printavec   = True
    printvol    = True
    printtensor = True
    printpos    = True
    printelas   = True
    
    #default values for existing parameters
    alatfile = 0.0
    blatfile = 0.0
    clatfile = 0.0
    
    try:
        with open(printfilename,'r') as p:
            for i, line in enumerate(p):
                if 'alat' in line:
                    printavec = False
                    #store lattice parameters from file for later check
                    l = linecache.getline(printfilename,i+1).split()
                    alatfile = float(l[1])
                    l = linecache.getline(printfilename,i+2).split()
                    blatfile = float(l[1])
                    l = linecache.getline(printfilename,i+3).split()
                    clatfile = float(l[1])
                elif 'volume' in line:
                    printvol = False
                elif 'dielectric' in line:
                    printtensor = False
                elif 'atpos' in line:
                    printpos = False
                elif 'elastic' in line:
                    printelas = False

    except:
        print('Output file not found, generating one now...')
        
    # With file located, we will read the file and look for particular things based 
    # on our current directory.
    latvec  = 'length of vectors'
    volcell = 'volume of cell'
    numions = 'number of ions'
    iontype = 'ions per type'
    dietens = ' MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)'
    bectens = ' BORN EFFECTIVE CHARGES (in e, cummulative output)'
    elatens = ' SYMMETRIZED ELASTIC MODULI (kBar)'
    
    #determine what to read in when the infilename ends with OUTCAR
    times = 0
    if infilename.endswith('OUTCAR'):
        with open(infilename,'r') as f:
            for i, line in enumerate(f):
                # By default the last time the lattice parameters and volume are printed 
               
                #get lattice vectors
                if latvec in line and printvol == True :
                    data = linecache.getline(infilename,i+2).split()
                    if data != []:
                        alat = float(data[0])
                        blat = float(data[1])
                        clat = float(data[2])
                
                #get unit cell volume
                elif volcell in line and printvol == True:
                    data = linecache.getline(infilename,i+1).split()
                    vol  = float(data[len(data)-1])
            
                #get number of ions in the unit cell
                elif numions in line and printtensor == True:
                    data = linecache.getline(infilename,i+1).split()
                    natom = int(data[11])
                    
                elif iontype in line and printtensor == True:
                    data = linecache.getline(infilename,i+1).split()
                    itype = []
                    for j in range(4,len(data)):
                        itype = np.append(itype,int(data[j]))
                    
                #get dielectric tensor
                elif dietens in line and printtensor == True: 
                    data1 = linecache.getline('OUTCAR',i+3).split()
                    data2 = linecache.getline('OUTCAR',i+4).split()
                    data3 = linecache.getline('OUTCAR',i+5).split()
                    dietensor[0][0] = float(data1[0])
                    dietensor[0][1] = float(data1[1])
                    dietensor[0][2] = float(data1[2])
                    dietensor[1][0] = float(data2[0])
                    dietensor[1][1] = float(data2[1])
                    dietensor[1][2] = float(data2[2])
                    dietensor[2][0] = float(data3[0])
                    dietensor[2][1] = float(data3[1])
                    dietensor[2][2] = float(data3[2])
                    
                #get elastic tensor
                elif elatens in line and printelas == True:
                    data1 = linecache.getline('OUTCAR',i+4).split()
                    data2 = linecache.getline('OUTCAR',i+5).split()
                    data3 = linecache.getline('OUTCAR',i+6).split()
                    data4 = linecache.getline('OUTCAR',i+7).split()
                    data5 = linecache.getline('OUTCAR',i+8).split()
                    data6 = linecache.getline('OUTCAR',i+9).split()
                    if not float(data1[1]) >= 0.0 and float(data2[2]) >= 0.0 and float(data3[3]) >= 0.0:
                        #checks the first three diagonal components... that should be enough
                        print('ERROR: The elastic tensor contains negative diagonal elements!')
                        print('       Diagonal elements are...')
                        print('       %s' %float(data1[1]))
                        print('       %s' %float(data2[2]))
                        print('       %s' %float(data3[3]))
                        print('       %s' %float(data4[4]))
                        print('       %s' %float(data5[5]))
                        print('       %s' %float(data6[6]))
                        print('       Aborting calculation!')
                        print('Suggestion: Check the elastic tensor calculation since your material us not stable.')
                        sys.exit()  
                    elatensor[0][0] = float(data1[1])
                    elatensor[0][1] = float(data1[2])
                    elatensor[0][2] = float(data1[3])
                    elatensor[0][3] = float(data1[4])
                    elatensor[0][4] = float(data1[5])
                    elatensor[0][5] = float(data1[6])
                    elatensor[1][0] = float(data2[1])
                    elatensor[1][1] = float(data2[2])
                    elatensor[1][2] = float(data2[3])
                    elatensor[1][3] = float(data2[4])
                    elatensor[1][4] = float(data2[5])
                    elatensor[1][5] = float(data2[6])
                    elatensor[2][0] = float(data3[1])
                    elatensor[2][1] = float(data3[2])
                    elatensor[2][2] = float(data3[3])
                    elatensor[2][3] = float(data3[4])
                    elatensor[2][4] = float(data3[5])
                    elatensor[2][5] = float(data3[6])
                    elatensor[3][0] = float(data4[1])
                    elatensor[3][1] = float(data4[2])
                    elatensor[3][2] = float(data4[3])
                    elatensor[3][3] = float(data4[4])
                    elatensor[3][4] = float(data4[5])
                    elatensor[3][5] = float(data4[6])
                    elatensor[4][0] = float(data5[1])
                    elatensor[4][1] = float(data5[2])
                    elatensor[4][2] = float(data5[3])
                    elatensor[4][3] = float(data5[4])
                    elatensor[4][4] = float(data5[5])
                    elatensor[4][5] = float(data5[6])
                    elatensor[5][0] = float(data6[1])
                    elatensor[5][1] = float(data6[2])
                    elatensor[5][2] = float(data6[3])
                    elatensor[5][3] = float(data6[4])
                    elatensor[5][4] = float(data6[5])
                    elatensor[5][5] = float(data6[6])                    
                    
                #get born effective charge tensor
                elif bectens in line and printtensor == True:
                    bectensor = np.zeros(shape=(natom,3,3))
                    for j in range(natom):
                        data1 = linecache.getline(infilename,i+4+j*4).split()
                        data2 = linecache.getline(infilename,i+5+j*4).split()
                        data3 = linecache.getline(infilename,i+6+j*4).split()
                        bectensor[j][0][0] = float(data1[1])
                        bectensor[j][0][1] = float(data1[2])
                        bectensor[j][0][2] = float(data1[3])
                        bectensor[j][1][0] = float(data2[1])
                        bectensor[j][1][1] = float(data2[2])
                        bectensor[j][1][2] = float(data2[3])
                        bectensor[j][2][0] = float(data3[1])
                        bectensor[j][2][1] = float(data3[2])
                        bectensor[j][2][2] = float(data3[3])     
                        
                elif printpos == True and times == 0:
                    path = infilename.split('/')
                    newpath = ''
                    for i in range(len(path)-1):
                        newpath += path[i]+'/'
                    newpath+='POSCAR'     
                    times = 1
                    with open(newpath,'r') as f:
                        for i, line in enumerate(f):
                            if i == 5 :
                                atnames = line.split()
                            elif i == 6:
                                atmult = line.split()
                                                    
    """
    Now start print out of information to a seperate file. What is printed is 
    determined by the input file and what is already in the seperate file.
    """
                
    #open file
    f1= open(printfilename,'a')
    errortol = 1E-6
    
    #check values in file
    if np.abs(alatfile-alat) >= errortol and alatfile != 0.0 and alat !=0:
        print('ERROR: The a lattice parameter read from the outcar file %s, \n      is signifantly different than the one in the file %s. \n      Please check calculation.'%(alat,alatfile))
        sys.exit()
    if np.abs(blatfile-blat) >= errortol and blatfile != 0.0 and blat !=0:
        print('ERROR: The b lattice parameter read from the outcar file %s, \n      is signifantly different than the one in the file %s. \n      Please check calculation.'%(blat,blatfile))
        sys.exit()
    if np.abs(clatfile-clat) >= errortol and clatfile != 0.0 and clat !=0:
        print('ERROR: The c lattice parameter read from the outcar file %s, \n      is signifantly different than the one in the file %s. \n      Please check calculation.'%(clat,clatfile))
        sys.exit()
        
    
    #print data to the output file (if it doesn't already exist)
    if alat != 0 and printavec== True:
        f1.write('#Calculated data from VASP for TDEP\n')
        f1.write('alat %f\n'%alat)
        f1.write('blat %f\n'%blat)
        f1.write('clat %f\n'%clat)
    if vol != 0 and printvol ==True:
        f1.write('volume %f\n'%vol)
        f1.write('TMIN %s\n' %tmin)
        f1.write('TMAX %s\n' %tmax)
        f1.write('TSTEP %s\n' %tsteps)
    if printpos == True:
        atnamestring = ''
        for i in range(len(atnames)):
            for j in range(int(atmult[i])):
                atnamestring += atnames[i]+' '
        atnamestring = atnamestring[:-1]  #removes the last space in the string
        f1.write('atpos %s\n'%atnamestring)
    if natom != 0 and bectensor != [] and printtensor== True:
        f1.write('natom %i\n'%natom)    
        f1.write('dielectric %f %f %f %f %f %f %f %f %f\n'%(dietensor[0][0],dietensor[0][1],dietensor[0][2],
                                    dietensor[1][0],dietensor[1][1],dietensor[1][2],
                                    dietensor[2][0],dietensor[2][1],dietensor[2][2]))
        f1.write('atommul ')
        for i in range(len(itype)):
            f1.write('%i ' %itype[i])
        f1.write('\n')
        for i in range(natom):
            
            if i == 0:
                f1.write('bectensor %i %f %f %f %f %f %f %f %f %f\n'%(i+1,bectensor[i][0][0],bectensor[i][0][1],bectensor[i][0][2],
                                      bectensor[i][1][0],bectensor[i][1][1],bectensor[i][1][2],
                                      bectensor[i][2][0],bectensor[i][2][1],bectensor[i][2][2]))  
            else:
                f1.write('          %i %f %f %f %f %f %f %f %f %f\n'%(i+1,bectensor[i][0][0],bectensor[i][0][1],bectensor[i][0][2],
                                      bectensor[i][1][0],bectensor[i][1][1],bectensor[i][1][2],
                                      bectensor[i][2][0],bectensor[i][2][1],bectensor[i][2][2]))

        f1.write('elastic %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n'
                 %(elatensor[0][0],elatensor[0][1],elatensor[0][2],elatensor[0][3],elatensor[0][4],elatensor[0][5],
                   elatensor[1][0],elatensor[1][1],elatensor[1][2],elatensor[1][3],elatensor[1][4],elatensor[1][5],
                   elatensor[2][0],elatensor[2][1],elatensor[2][2],elatensor[2][3],elatensor[2][4],elatensor[2][5],
                   elatensor[3][0],elatensor[3][1],elatensor[3][2],elatensor[3][3],elatensor[3][4],elatensor[3][5],
                   elatensor[4][0],elatensor[4][1],elatensor[4][2],elatensor[4][3],elatensor[4][4],elatensor[4][5],
                   elatensor[5][0],elatensor[5][1],elatensor[5][2],elatensor[5][3],elatensor[5][4],elatensor[5][5]))
                                
    #close output filein
    f1.close()
    return None

def bash_commands(bashcommand):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Execute bash command and output any error message
    
    Return: string with a message (if any) about an error in the script
    """
    output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True).communicate()[0].decode('utf-8').strip()
    
    return str(output)

def bash_commands_cwd(bashcommand,wd):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Execute bash command and output any error message
    
    Return: string with a message (if any) about an error in the script
    """
    output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True,cwd=wd).communicate()[0].decode('utf-8').strip('\n')
    
    return str(output)

def make_folder(folder_name):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: execute bash command to generate a new folder called "folder_name"
        
    """
    bashcommand = 'mkdir '+folder_name
    output = bash_commands(bashcommand)
    
    return output

def move_file(filename,newlocation,original):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: To move a file to a new folder, keeping a copy of that file in the orginal folder if original == True 
    """
    
    if original == True:
        bashcp = 'cp '+filename+' MOVEDCOPY'
        output = bash_commands(bashcp)
        
        bashmv = 'mv MOVEDCOPY '+newlocation+'/'+filename
        output = bash_commands(bashmv)
        
    else:
        bashmv = 'mv '+filename+' '+newlocation+'/'+filename
        output = bash_commands(bashmv)
        
    return output

def remove_file(filename):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Remove a file 
    """
    
    bashrm = 'rm '+filename
    output = bash_commands(bashrm)
    
    return output

def copy_file(originalfile,newfile):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: To move a file to a new folder, keeping a copy of that file in the orginal folder if original == True 
    """

    bashcp = 'cp '+originalfile+' '+newfile
    output = bash_commands(bashcp)
        
    return output

def get_batchheader(batchtype):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Generate the batch header file for submission to different computer 
             systems. (slurm or pbs)
    
    Return: A string with the header for submission scripts
    """
    batch_header = ''
    if supercomputer_software == "slurm":
        if batchtype == 'Relaxation':
            batch_header =  '# Specify jobname:\n'\
                            '#SBATCH --job-name=rlx\n'\
                            '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                            '#SBATCH --nodes=1  --ntasks-per-node=16\n'\
                            '#SBATCH --account='+str(account_number)+'\n'\
                            '# The maximum time allowed for the job, in hh:mm:ss\n'\
                            '#SBATCH --time=3:00:00\n'\
                            '#SBATCH --mem-per-cpu=1800M\n'\
                            '#SBATCH --mail-user='+str(account_email)+'\n'\
                            '#SBATCH --mail-type=ALL'
        elif batchtype ==  'Elastic':
            batch_header =  '# Specify jobname:\n'\
                            '#SBATCH --job-name=elas\n'\
                            '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                            '#SBATCH --nodes=2  --ntasks-per-node=16\n'\
                            '#SBATCH --account='+str(account_number)+'\n'\
                            '# The maximum time allowed for the job, in hh:mm:ss\n'\
                            '#SBATCH --time=15:00:00\n'\
                            '#SBATCH --mem-per-cpu=1800M\n'\
                            '#SBATCH --mail-user='+str(account_email)+'\n'\
                            '#SBATCH --mail-type=ALL'                    
        elif batchtype == 'TDEP' or batchtype == 'TDEP2':
            batch_header = '# Specify jobname:\n'\
                            '#SBATCH --job-name=tdep\n'\
                            '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                            '#SBATCH --nodes=2  --ntasks-per-node=16\n'\
                            '#SBATCH --account='+str(account_number)+'\n'\
                            '# The maximum time allowed for the job, in hh:mm:ss\n'\
                            '#SBATCH --time=4:00:00\n'\
                            '#SBATCH --mem-per-cpu=2000M\n'\
                            '#SBATCH --mail-user='+str(account_email)+'\n'\
                            '#SBATCH --mail-type=FAIL'    #Too many emails if you use ALL  
        elif batchtype == 'Ground_state':
            batch_header  =  '# Specify jobname:\n'\
                            '#SBATCH --job-name=tdepconfig\n'\
                            '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                            '#SBATCH --nodes=4  --ntasks-per-node=16\n'\
                            '#SBATCH --account='+str(account_number)+'\n'\
                            '# The maximum time allowed for the job, in hh:mm:ss\n'\
                            '#SBATCH --time=12:00:00\n'\
                            '#SBATCH --mem-per-cpu=1800M\n'\
                            '#SBATCH --mail-user='+str(account_email)+'\n'\
                            '#SBATCH --mail-type=FAIL'  
    
    #elif supercomputer_software == 'pbs':
        
    else:
        print('ERROR: Supercomputer submission not supported.  Contact Developer.')
        sys.exit()
    
    
    return batch_header

def generate_batch(batchtype,batchname,tags):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Automatic generation of the submission script for each HPC system
             and for each calculation type. The batch script is printed to the correct 
             folder for each calculation.

             batchtype - relax, elastic, etc
             batchname - name of batchfile
             
    Return: None
    """
    #unpack tags
    if tags != '':
        #withbounds = tags[0]
        withsolver = tags[1]
        withBEC    = tags[2]
    else:
        withBEC    = False
        withsolver = False
    
    batch_header = get_batchheader(batchtype)
    
    #generate scripts
    if batchtype == 'Relaxation' :  
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header)
        f.write('\n')
        f.write(VASPSR+'\n')
        f.write(PYTHMOD+'\n')
        f.write('\n')
        f.write('## submit vasp job\n')
        f.write('if [ -f "OUTCAR" ] && [ `python ../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write(' echo "Relaxing structure again..."\n') 
        f.write('  mv CONTCAR POSCAR\n')        
        f.write('  $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('  if [ `python ../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('   echo "Converged on second run."\n')
        f.write('   python ../../ACTE.py --outcar\n')
        f.write('   echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('   python ../../ACTE.py --copy_file CONTCAR CONTCAR2\n')
        f.write('   python ../../ACTE.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('   python ../../ACTE.py --generate_batch Elastic batch.sh\n')
        f.write('   python ../../ACTE.py --move_file batch.sh ../Elastic False\n')
        f.write('   python ../../ACTE.py --launch_calc ../Elastic batch.sh\n')
        f.write('   else\n') 
        f.write('   echo "Run failed to converge. Aborting"\n')
        f.write('   exit 1 \n')
        f.write('   fi\n')
        f.write('else\n')
        f.write(' echo "Starting first relaxation."\n')
        f.write(' $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write(' rm -f WAVECAR CHG\n')
        f.write(' if [ `python ../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('  echo "Converged in first attempt, Relaxing again..."\n')
        f.write('  mv CONTCAR POSCAR\n')
        f.write('  echo "Starting second relaxation."\n')
        f.write('  $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('  if [ `python ../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('   echo "Converged on second run, moving on"\n')
        f.write('   python ../../ACTE.py --outcar\n')
        f.write('   echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('   python ../../ACTE.py --copy_file CONTCAR CONTCAR2\n')
        f.write('   python ../../ACTE.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('   python ../../ACTE.py --generate_batch Elastic batch.sh\n')
        f.write('   python ../../ACTE.py --move_file batch.sh ../Elastic False\n')
        f.write('   python ../../ACTE.py --launch_calc ../Elastic batch.sh\n')
        f.write('  else\n')        
        f.write('   mv CONTCAR POSCAR\n')
        f.write('   echo "Starting third relaxation."\n')        
        f.write('   $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('   rm -f WAVECAR CHG\n')
        f.write('   if [ `python ../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('    echo "Converged on third run, moving on"\n')
        f.write('    python ../../ACTE.py --outcar\n')
        f.write('    echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('    python ../../ACTE.py --copy_file CONTCAR CONTCAR2\n')
        f.write('    python ../../ACTE.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('    python ../../ACTE.py --generate_batch Elastic batch.sh\n')
        f.write('    python ../../ACTE.py --move_file batch.sh ../Elastic False\n')
        f.write('    python ../../ACTE.py --launch_calc ../Elastic batch.sh\n')
        f.write('   else\n') 
        f.write('    echo "Run failed to converge after 3 attempts. Aborting"\n')
        f.write('    exit 1 \n')
        f.write('   fi\n')
        f.write('  fi\n')
        f.write(' else\n')
        f.write('  mv CONTCAR POSCAR\n')
        f.write('  echo "Restart relaxation."\n')
        f.write('  $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('  rm -f WAVECAR CHG\n')
        f.write('  if [ `python ../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('   echo "Converged on second run, moving on"\n')
        f.write('   python ../../ACTE.py --outcar\n')
        f.write('   echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('   python ../../ACTE.py --copy_file CONTCAR CONTCAR2\n')
        f.write('   python ../../ACTE.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('   python ../../ACTE.py --generate_batch Elastic batch.sh\n')
        f.write('   python ../../ACTE.py --move_file batch.sh ../Elastic False\n')
        f.write('   python ../../ACTE.py --launch_calc ../Elastic batch.sh\n')
        f.write('  else\n')
        f.write('   mv CONTCAR POSCAR\n')
        f.write('   echo "Starting third relaxation."\n')        
        f.write('   $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('   rm -f WAVECAR CHG\n')
        f.write('   if [ `python ../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('    echo "Converged on third run, moving on"\n')
        f.write('    python ../../ACTE.py --outcar\n')
        f.write('    echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('    python ../../ACTE.py --copy_file CONTCAR CONTCAR2\n')
        f.write('    python ../../ACTE.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('    python ../../ACTE.py --generate_batch Elastic batch.sh\n')
        f.write('    python ../../ACTE.py --move_file batch.sh ../Elastic False\n')
        f.write('    python ../../ACTE.py --launch_calc ../Elastic batch.sh\n')
        f.write('   else\n') 
        f.write('    echo "Run failed to converge after 3 attempts. Aborting"\n')
        f.write('    exit 1 \n')
        f.write('   fi\n')
        f.write('  fi\n')
        f.write(' fi\n')
        f.write('fi\n')
        f.close()
               
    elif batchtype == 'Elastic':
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header)
        f.write('\n')
        f.write(VASPSR)
        f.write('\n')
        f.write(PYTHMOD+'\n')
        f.write('## submit vasp job\n')
        f.write(' mv CONTCAR2 POSCAR\n')
        f.write('$MPIEXEC_LOCAL $VASPLOC $option\n')     
        f.write('   python ../../ACTE.py --outcar\n')     
        f.write('\n')
        f.write('#Automatically launch lattice calculations\n')
        f.write('   cd ../\n')
        f.write('   python ../ACTE.py --build_cells ')
        f.close()
        
    elif batchtype == 'TDEP':
        #gather information from data_extract file
        debye = 0
        with open('data_extraction','r') as datafile:
            for line in datafile:
                if 'Debye' in line:
                    l = line.split()
                    debye = l[1]
        
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header)
        f.write('\n')
        f.write(VASPSR)
        f.write('\n')
        f.write(PYTHMOD+'\n')
        f.write('#set up directories and files\n')
        f.write('rlx_dir=relaxation2\n')
        f.write('initial_MD_dir=moldyn\n')
        f.write('config_dir=configs\n')
        f.write('# Default parameters:\n')
        f.write('natom='+natom_ss+'\n')
        f.write('n_configs='+n_configs+'\n')
        f.write('t_configs='+str(float(t_configs)*float(debye))+'\n')
        f.write('debye=%s\n'%debye)
        f.write('\n')
        f.write('echo "start relaxation of cell after cell parameter change"\n')
        f.write('mv POSCAR_* POSCAR\n')
        f.write('   echo "Launching refine structure to resymmetrize the cell"\n')
        f.write('   mv POSCAR infile.ucposcar\n' )
        f.write('   srun -n 1 '+TDEPSR+'refine_structure\n')
        f.write('   mv outfile.refined_cell POSCAR\n' )
        f.write('mkdir -p $rlx_dir\n')
        f.write('cp -up POSCAR $rlx_dir/\n')
        f.write('cp -up POTCAR $rlx_dir/  #use previously generated POTCAR\n')
        f.write('cp -up INCAR_rlx $rlx_dir/INCAR\n')
        f.write('cp -up KPOINTS $rlx_dir/\n')
        f.write('\n')
        f.write('cd $rlx_dir\n')
        f.write('if [ -s "OUTCAR" ] && [ `python ../../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('    echo "Already converged, moving on"\n')
        f.write('else\n')
        f.write('\n')
        f.write('   #create last file and run vasp  \n')
        f.write('   $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('   rm -f WAVECAR CHG\n')
        f.write('   echo "Launching refine structure to resymmetrize the cell"\n')
        f.write('   mv POSCAR infile.ucposcar\n' )
        f.write('   srun -n 1 '+TDEPSR+'refine_structure\n')
        f.write('   mv outfile.refined_cell POSCAR\n' )
        f.write('fi\n')
        f.write('\n')
        f.write('cd ..\n')
        f.write('echo "Starting configuration creation "\n')
        f.write('\n')
        f.write('mkdir -p $initial_MD_dir\n')
        f.write('cp -up INCAR $initial_MD_dir/INCAR\n')
        f.write('cp -up $rlx_dir/CONTCAR $initial_MD_dir/infile.ucposcar\n')
        f.write('cp -up POTCAR $initial_MD_dir/\n')
        f.write('cp -up ../infile.lotosplitting $initial_MD_dir/\n')
        f.write('\n')
        f.write('cd  $initial_MD_dir/\n')
        f.write('# Create start structure for high accuracy calcs at finite temperature \n')
        f.write('if [ -f "infile.forceconstant" ]; then\n')
        f.write('    echo "Already found force constants, moving on"\n')
        f.write('else\n')
        f.write('    echo "Run generate_structure" \n')
        f.write('   srun -n 1  '+TDEPSR+'generate_structure -na $natom\n')
        f.write('\n')
        f.write('     ln -s outfile.ssposcar infile.ssposcar\n')
        f.write('     cp outfile.ssposcar POSCAR\n')
        f.write('fi\n')
        f.write('\n')
        f.write('if [ ! -f "contcar_conf0001" ]; then\n')
        f.write('    echo "Run canonical_configuration"\n')
        f.write('      srun -n 1 '+TDEPSR+'canonical_configuration -n $n_configs -t $t_configs -td $debye --quantum\n')
        f.write('  fi\n')
        f.write(' cp outfile.fakeforceconstant infile.forceconstant\n')
        f.write('\n')
        f.write('if [ ! -f "contcar_conf0001" ]; then\n')
        f.write('    echo "canonical_configuration failed; cannot find contcar_conf0001. Exiting."\n')
        f.write('    exit 1\n')
        f.write('fi\n')
        f.write('echo "Run file operations for configs first iteration"\n')
        f.write('mkdir -p ../$config_dir\n')
        f.write('mv -u contcar_* ../$config_dir\n')
        f.write('cp -up infile.ucposcar outfile.ssposcar infile.lotosplitting POTCAR ../$config_dir\n')
        f.write('cp -up INCAR ../$config_dir/INCAR\n')
        f.write('cd ../$config_dir\n')
        f.write('\n')
        f.write('echo "Current dir: " $PWD\n')
        f.write('if [ -f "outfile.free_energy" ] ; then\n')
        f.write('    rm -f contcar_*\n')
        f.write('    echo "Calculation already converged, continuing"\n')
        f.write('else\n')
        f.write('    g="contcar_"\n')
        f.write('    for f in contcar_conf*; do\n')
        f.write('        if [ -e "$f" ]; then\n')
        f.write('            echo $f\n')
        f.write('        else\n')
        f.write('            echo "configurations do not exist, exiting"; exit 1\n')
        f.write('        fi\n')
        f.write('        dir=${f#$g}\n')
        f.write('        mkdir -p $dir\n')
        f.write('        if [ -e "$dir/POSCAR" ]; then\n')
        f.write('            rm -f $f\n')
        f.write('            echo "$dir/POSCAR already exists, skipping"\n')
        f.write('        else\n')
        f.write('            mv -n $f $dir/POSCAR\n')
        f.write('        fi\n')
        f.write('    done\n')
        f.write('    ln -s outfile.ssposcar infile.ssposcar\n')
        f.write('\n')
        f.write('# Build VASP files\n')
        f.write('    echo "Build VASP configs "\n')
        f.write('    for d in conf*; do\n')
        f.write('    cd $d\n')
        f.write('    if [ -f "OUTCAR" ] ; then\n')
        f.write('       pythonresult=$(python ../../../../ACTE.py --vasp_converge TDEP)\n')
        f.write('    fi\n')
        f.write('    cd ..\n')
        f.write('    if [ -f "$d/OUTCAR" ] && [ "$pythonresult" == "True" ] ; then\n')
        f.write('            echo "Already converged in $d, moving on"\n')
        f.write('        else\n')
        f.write('            cp -up INCAR POTCAR $d\n')
        f.write('            cd $d\n')
        f.write('            #make KPOINT File\n')
        f.write('            python ../../../../ACTE.py --make_KPOINTS '+str(int(2.0*kdensity))+'\n')
        f.write('            python ../../../../ACTE.py --generate_batch Ground_state batch.sh \n')
        f.write('            #$MPIEXEC_LOCAL $VASPLOC $option\n') 
        f.write('            #rm -f CHG WAVECAR DOSCAR CHGCAR vasprun.xml\n')
        f.write('            cd ..\n')
        f.write('            python ../../../ACTE.py --launch_calc $d batch.sh \n')
        f.write('        fi\n')
        f.write('    done\n')
        f.write('\n')
        f.write('fi\n')
        f.write('\n')
        f.write('echo "Generation of configurations complete. "\n')
        f.write('\n')
        f.close()
    
    elif batchtype == 'TDEP2':  
        #gather information from data_extract file
        debye = 0
        with open('data_extraction','r') as datafile:
            for line in datafile:
                if 'Debye' in line:
                    l = line.split()
                    debye = l[1]
        
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header)
        f.write('\n')
        f.write(VASPSR)
        f.write('\n')
        f.write(PYTHMOD+'\n')
        f.write('#set up directories and files\n')
        f.write('rlx_dir=relaxation2\n')
        f.write('initial_MD_dir=moldyn\n')
        f.write('config_dir=configs\n')
        f.write('# Default parameters:\n')
        f.write('natom='+natom_ss+'\n')
        f.write('n_configs='+n_configs+'\n')
        f.write('t_configs='+str(float(t_configs)*float(debye))+'\n')
        f.write('\n')
        f.write('echo "start relaxation of cell after cell parameter change"\n')
        f.write('mv POSCAR_* POSCAR\n')
        f.write('mkdir -p $rlx_dir\n')
        f.write('cp -up POSCAR $rlx_dir/\n')
        f.write('cp -up POTCAR $rlx_dir/  #use previously generated POTCAR\n')
        f.write('cp -up INCAR_rlx $rlx_dir/INCAR\n')
        f.write('cp -up KPOINTS $rlx_dir/\n')
        f.write('\n')
        f.write('cd $rlx_dir\n')
        f.write('if [ -s "OUTCAR" ] && [ `python ../../../ACTE.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('    echo "Already converged, moving on"\n')
        f.write('else\n')
        f.write('\n')
        f.write('   #create last file and run vasp  \n')
        f.write('   $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('   rm -f WAVECAR CHG\n')
        f.write('   echo "Launching refine structure to resymmetrize the cell"\n')
        f.write('   mv POSCAR infile.ucposcar\n' )
        f.write('   refine_structure\n')
        f.write('   mv outfile.refined_cell POSCAR\n' )
        f.write('fi\n')
        f.write('\n')
        f.write('cd ..\n')
        f.write('echo "Starting configuration creation "\n')
        f.write('\n')
        f.write('mkdir -p $initial_MD_dir\n')
        f.write('cp -up INCAR $initial_MD_dir/INCAR\n')
        f.write('cp -up $rlx_dir/CONTCAR $initial_MD_dir/infile.ucposcar\n')
        f.write('cp -up POTCAR $initial_MD_dir/\n')
        f.write('cp -up ../infile.lotosplitting $initial_MD_dir/\n')
        f.write('\n')
        f.write('cd  $initial_MD_dir/\n')
        f.write('# Create start structure for high accuracy calcs at finite temperature \n')
        f.write('if [ -f "infile.forceconstant" ]; then\n')
        f.write('    echo "Already found force constants, moving on"\n')
        f.write('else\n')
        f.write('    echo "Run generate_structure" \n')
        f.write('   srun -n 1  '+TDEPSR+'generate_structure -na $natom\n')
        f.write('\n')
        f.write('     ln -s outfile.ssposcar infile.ssposcar\n')
        f.write('     cp outfile.ssposcar POSCAR\n')
        f.write('fi\n')
        f.write('\n')
        f.write('if [ ! -f "contcar_conf0001" ]; then\n')
        f.write('    echo "Run canonical_configuration"\n')
        f.write('      srun -n 1 '+TDEPSR+'canonical_configuration -n $n_configs -t $t_configs --quantum\n')
        f.write('  fi\n')
        f.write(' cp outfile.fakeforceconstant infile.forceconstant\n')
        f.write('\n')
        f.write('if [ ! -f "contcar_conf0001" ]; then\n')
        f.write('    echo "canonical_configuration failed; cannot find contcar_conf0001. Exiting."\n')
        f.write('    exit 1\n')
        f.write('fi\n')
        f.write('echo "Run file operations for configs first iteration"\n')
        f.write('mkdir -p ../$config_dir\n')
        f.write('mv -u contcar_* ../$config_dir\n')
        f.write('cp -up infile.ucposcar outfile.ssposcar infile.lotosplitting POTCAR ../$config_dir\n')
        f.write('cp -up INCAR ../$config_dir/INCAR\n')
        f.write('cd ../$config_dir\n')
        f.write('\n')
        f.write('echo "Current dir: " $PWD\n')
        f.write('if [ -f "outfile.free_energy" ] ; then\n')
        f.write('    rm -f contcar_*\n')
        f.write('    echo "Calculation already converged, continuing"\n')
        f.write('else\n')
        f.write('    g="contcar_"\n')
        f.write('    for f in contcar_conf*; do\n')
        f.write('        if [ -e "$f" ]; then\n')
        f.write('            echo $f\n')
        f.write('        else\n')
        f.write('            echo "configurations do not exist, exiting"; exit 1\n')
        f.write('        fi\n')
        f.write('        dir=${f#$g}\n')
        f.write('        mkdir -p $dir\n')
        f.write('        if [ -e "$dir/POSCAR" ]; then\n')
        f.write('            rm -f $f\n')
        f.write('            echo "$dir/POSCAR already exists, skipping"\n')
        f.write('        else\n')
        f.write('            mv -n $f $dir/POSCAR\n')
        f.write('        fi\n')
        f.write('    done\n')
        f.write('    ln -s outfile.ssposcar infile.ssposcar\n')
        f.write('\n')
        f.write('# Build VASP files\n')
        f.write('    echo "Build VASP configs "\n')
        f.write('    for d in conf*; do\n')
        f.write('    cd $d\n')
        f.write('    if [ -f "OUTCAR" ] ; then\n')
        f.write('       pythonresult=$(python ../../../../ACTE.py --vasp_converge TDEP)\n')
        f.write('    fi\n')
        f.write('    cd ..\n')
        f.write('    if [ -f "$d/OUTCAR" ] && [ "$pythonresult" == "True" ] ; then\n')
        f.write('            echo "Already converged in $d, moving on"\n')
        f.write('        else\n')
        f.write('            cp -up INCAR POTCAR $d\n')
        f.write('            cd $d\n')
        f.write('            #make KPOINT File\n')
        f.write('            python ../../../../ACTE.py --make_KPOINTS '+str(int(2.0*kdensity))+'\n')
        f.write('            python ../../../../ACTE.py --generate_batch Ground_state batch.sh \n')
        f.write('            #$MPIEXEC_LOCAL $VASPLOC $option\n') 
        f.write('            #rm -f CHG WAVECAR DOSCAR CHGCAR vasprun.xml\n')
        f.write('            cd ..\n')
        f.write('            python ../../../ACTE.py --launch_calc $d batch.sh \n')
        f.write('        fi\n')
        f.write('    done\n')
        f.write('\n')
        f.write('fi\n')
        f.write('\n')
        f.write('echo "Generation of configurations complete. "\n')
        f.write('\n')
        f.close()
        
    elif batchtype == 'Ground_state':
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header)
        f.write('\n')
        f.write(VASPSR)
        f.write('\n')
        f.write(PYTHMOD+'\n')
        if use_scratch == 'yes':
            f.write('# Define and create a unique scratch directory for this job\n')
            f.write('SCRATCH_DIRECTORY=/global/work/${USER}/${SLURM_JOBID}.stallo-adm.uit.no\n')
            f.write('mkdir -p ${SCRATCH_DIRECTORY}\n')
            f.write('cd ${SCRATCH_DIRECTORY}\n')
            f.write('# You can copy everything you need to the scratch directory\n')
            f.write('cp ${SLURM_SUBMIT_DIR}/* ${SCRATCH_DIRECTORY}\n')
        f.write('## submit vasp job\n')
        f.write('$MPIEXEC_LOCAL $VASPLOC $option\n')     
        f.write('rm -f CHG WAVECAR DOSCAR CHGCAR vasprun.xml\n')
        if use_scratch == 'yes':
            f.write('# copy back all data after calculation runs\n')
            f.write('cp ${SCRATCH_DIRECTORY}/* ${SLURM_SUBMIT_DIR}\n')
            f.write('cd ${SLURM_SUBMIT_DIR}\n')
            f.write('rm -rf ${SCRATCH_DIRECTORY}\n')
        f.close()
        
    elif batchtype == 'script':
        #print file
        f = open(batchname,'w')
        f.write('#loop through configuration files and run TDEP \n')
        f.write(PYTHMOD+'\n')
        f.write('python2 '+TDEPSR+'process_outcar_5.3.py */OUTCAR\n')
        if withsolver == True and withBEC == True:
            f.write(TDEPSR+'extract_forceconstants -rc2 '+rc_cut+' --polar -U0 --solver 2\n')
        elif withsolver == False and withBEC == True:
            f.write(TDEPSR+'extract_forceconstants -rc2 '+rc_cut+' --polar -U0 --solver 1\n')
        elif withsolver == True and withBEC == False:
            f.write(TDEPSR+'extract_forceconstants -rc2 '+rc_cut+' -U0 --solver 2\n')
        elif withsolver == False and withBEC == False:
            f.write(TDEPSR+'extract_forceconstants -rc2 '+rc_cut+' -U0 --solver 1\n')
        f.write('ln -s outfile.forceconstant infile.forceconstant\n')
        f.write(TDEPSR+'phonon_dispersion_relations --dos --temperature_range '+tmin+' '+tmax+' '+tsteps+' -qg '+qgrid+' -it '+iter_type+' --unit icm\n')    
        f.close()
        
        
    #make batch script executable
    bashsub = 'chmod +x '+batchname
    outp = bash_commands(bashsub)
    
    outp = outp #removes warning message
        
    return None

def generate_INCAR(incartype):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@smn.uio.no
    
    Purpose: Generate INCAR file for vasp calculations using some predefined 
             quantities and convergence parameters. The generated incar file is 
             placed in the correct folder at the time of its creation.
             
    Return: None
    """
    name  = 'INCAR'
    ECUT  = 'ENCUT = '+ecut+'\n'
    EDIFF = 'EDIFF = '+ediff+'\n'
        
    if incartype == 'Relaxation':
        #print file
        f = open(name,'w') #printed to an internal directory
        f.write('INCAR for ionic relaxation   (AUTOMATICALLY GENERATED)\n')
        f.write('\n')
        f.write('! Electronic relaxation\n')
        f.write('IALGO   = 48      ! Algorithm for electronic relaxation\n')
        f.write('NELMIN = 4         ! Minimum # of electronic steps\n')
        f.write(EDIFF)
        f.write(ECUT)
        f.write('PREC   = Accurate  ! Normal/Accurate\n')
        f.write('LREAL  = Auto      ! Projection in reciprocal space?\n')
        f.write('ISMEAR = 1         ! Smearing of partial occupancies. Metals: 1; else < 1.\n')
        f.write('SIGMA  = 0.2       ! Smearing width\n')
        f.write('ISPIN  = 1         ! Spin polarization? 1-no 2- yes\n')
        f.write('\n')
        f.write('! Ionic relaxation\n')
        f.write('EDIFFG = -0.0005     ! Tolerance for ions\n')
        f.write('NSW    = 800        ! Max # of ionic steps\n')
        f.write('MAXMIX = 80        ! Keep dielectric function between ionic movements\n')
        f.write('IBRION = 2         ! Algorithm for ions. 0: MD 1: QN/DIIS 2: CG\n')
        f.write('ISIF   = 3         ! Relaxation. 2: ions 3: ions+cell\n')
        f.write('ADDGRID= .TRUE.    ! More accurate forces with PAW\n')
        f.write('\n')
        f.write('! Output options\n')
        f.write('NWRITE = 1         ! Write electronic convergence at first step only\n')
        f.write('\n')
        f.write('! Memory handling\n')
        f.write('LPLANE  = .TRUE.\n')
        f.write('LSCALU  = .FALSE\n')
        f.write('NSIM    = 4\n')
        f.write('NCORE  = 4\n')
        f.write('LREAL=.FALSE.\n')
        f.write('\n')
        f.close()
               
    elif incartype == 'Elastic':
        #print file
        f1 = open(name,'w') #printed to an internal directory
        f1.write('INCAR for elastic tensor (AUTOMATICALLY GENERATED)\n')
        f1.write('! Electronic relaxation\n')
        f1.write('ALGO   = Fast      ! Algorithm for electronic relaxation\n')
        f1.write('NELMIN = 4         ! Minimum # of electronic steps\n')
        f1.write('NELM   = 200       ! Maximum # of electronic steps\n')
        f1.write(ECUT)
        f1.write(EDIFF)
        f1.write('PREC   = Accurate  ! Normal/Accurate\n')
        f1.write('LREAL  = .FALSE.      ! Projection in reciprocal space?\n')
        f1.write('ISMEAR = -5         ! Smearing of partial occupancies. k-points >2: -5; else 0\n')
        f1.write('SIGMA  = 0.2       ! Smearing width\n')
        f1.write('ISPIN  = 1         ! Spin polarization?\n')
        f1.write('\n')
        f1.write('## calculation of the q=0 phonon modes and eigenvectors.  Elastic tensor also calculated\n')
        f1.write('IBRION = 6\n')
        f1.write('NFREE  = 2\n')
        f1.write('ISIF = 3\n')
        f1.write('LEPSILON = .TRUE.\n')
        f1.write('! Output options\n')
        f1.write('NWRITE = 1         ! Write electronic convergence at first step only\n')
        f1.write('\n')
        f1.close()
        
    elif incartype == 'TDEP':
        #print file
        f1 = open(name,'w') #printed to an internal directory
        f1.write('INCAR for molecular dynamics\n')
        f1.write('\n')
        f1.write('! Electronic relaxation\n')
        f1.write('ALGO   = Fast      ! Algorithm for electronic relaxation\n')
        f1.write('NELMIN = 4         ! Minimum # of electronic steps\n')
        f1.write('NELM   = 500       ! Maximum # of electronic steps\n')
        f1.write(EDIFF)
        f1.write(ECUT)
        f1.write('PREC   = High      ! Normal/Accurate\n')
        f1.write('LREAL  = .False.   ! Projection in reciprocal space? False gives higher accuracy\n')
        f1.write('ISMEAR = -5         ! Smearing of partial occupancies. k-points >2: -5; else 0\n')
        f1.write('SIGMA  = 0.2       ! Smearing width\n')
        f1.write('ISPIN  = 1         ! Spin polarization?\n')
        f1.write('\n')
        f1.write('! Ionic relaxation\n')
        f1.write('NSW    = 0         ! Number of MD steps\n')
        f1.write('ISIF    = 2\n')
        f1.write('\n')
        f1.write('! Output options\n')
        f1.write('!LWAVE  = .FALSE.   ! Write WAVECAR?\n')
        f1.write('!LCHARG = .FALSE.   ! Write CHGCAR?\n')
        f1.write('NWRITE = 1         ! Write electronic convergence etc. at first and last step only\n')
        f1.write('\n')
        f1.write('! Memory handling\n')
        f1.write('NCORE   = 8\n')
        f1.close()
        
    elif incartype == 'TDEP_rlx':
        #print file
        f = open(name,'w') #printed to an internal directory
        f.write('INCAR for ionic relaxation   (AUTOMATICALLY GENERATED)\n')
        f.write('\n')
        f.write('! Electronic relaxation\n')
        f.write('IALGO   = 48      ! Algorithm for electronic relaxation\n')
        f.write('NELMIN = 4         ! Minimum # of electronic steps\n')
        f.write(EDIFF)
        f.write(ECUT)
        f.write('PREC   = Accurate  ! Normal/Accurate\n')
        f.write('LREAL  = Auto      ! Projection in reciprocal space?\n')
        f.write('ISMEAR = 1         ! Smearing of partial occupancies. Metals: 1; else < 1.\n')
        f.write('SIGMA  = 0.2       ! Smearing width\n')
        f.write('ISPIN  = 1         ! Spin polarization? 1-no 2- yes\n')
        f.write('\n')
        f.write('! Ionic relaxation\n')
        f.write('EDIFFG = -0.0005     ! Tolerance for ions\n')
        f.write('NSW    = 800        ! Max # of ionic steps\n')
        f.write('MAXMIX = 80        ! Keep dielectric function between ionic movements\n')
        f.write('IBRION = 1         ! Algorithm for ions. 0: MD 1: QN/DIIS 2: CG\n')
        f.write('ISIF   = 2         ! Relaxation. 2: ions 3: ions+cell\n')
        f.write('ADDGRID= .TRUE.    ! More accurate forces with PAW\n')
        f.write('\n')
        f.write('! Output options\n')
        f.write('NWRITE = 1         ! Write electronic convergence at first step only\n')
        f.write('\n')
        f.write('! Memory handling\n')
        f.write('LPLANE  = .TRUE.\n')
        f.write('LSCALU  = .FALSE\n')
        f.write('NSIM    = 4\n')
        f.write('NCORE  = 4\n')
        f.write('LREAL=.FALSE.\n')
        f.write('\n')
        f.close()
                   
    else:
        print('INCAR type not programmed. Aborting.')
        sys.exit()
        
    return None

def generate_POTCAR():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: generate potcar file using the poscar file to read in the atom types
    """
    #open POSCAR for reading
    
    infile = open('POSCAR','r') #open file for reading
    
    infile.readline() #comment line
    infile.readline() #scale 
    infile.readline() #first vector
    infile.readline() #second vector
    infile.readline() #third vector
    atomnames = infile.readline()
    
    atom_name = atomnames.split()
    
    string = ''
    
    for i in range(len(atom_name)):
        POTCARfound = False
        if os.path.isfile('../pseudos/'+atom_name[i]+'/POTCAR') and POTCARfound == False:
            string += '../pseudos/'+atom_name[i]+'/POTCAR '
            print('Using POTCAR with no extension for %s'%atom_name[i])
            POTCARfound = True
        elif os.path.isfile('../pseudos/'+atom_name[i]+'_sv/POTCAR') and POTCARfound == False:
            string += '../pseudos/'+atom_name[i]+'_sv/POTCAR ' 
            print('Using POTCAR with sv extension for %s'%atom_name[i])
            POTCARfound = True
        elif os.path.isfile('../pseudos/'+atom_name[i]+'_s/POTCAR') and POTCARfound == False:
             string += '../pseudos/'+atom_name[i]+'_s/POTCAR ' 
             print('Using POTCAR with s extension for %s'%atom_name[i])
             POTCARfound = True            
        else:
             print('ERROR: POTCAR not found for %s.' %atom_name[i])
             sys.exit()
              
    #create POTCAR file
    bashpotcar = 'cat '+string+' > POTCAR'
    output = bash_commands(bashpotcar) 
    
    output = output  #removes warning
    
    return None

def generate_KPOINT(kdensity):
    """
    Author:Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Generate K-point file using the specified kpoint density
    
    """
    kdist=1.0/float(kdensity)

    #read  POSCAR 
    infile = open('POSCAR', 'r')  # open file for reading 

    infile.readline() #comment line
    scaleline = infile.readline()
    scale = float(scaleline)

    vec1line = infile.readline()
    vec2line = infile.readline()
    vec3line = infile.readline()
    
    #determine quantities from POSCAR file
    vec1 = []; vec2 = []; vec3 = []
    vec1 = np.array(vec1line.split(), dtype=np.float32)
    vec2 = np.array(vec2line.split(), dtype=np.float32)
    vec3 = np.array(vec3line.split(), dtype=np.float32)

    alength = scale*np.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1] + vec1[2]*vec1[2])
    blength = scale*np.sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1] + vec2[2]*vec2[2])
    clength = scale*np.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
    
	# Calculate required density of k-points
    nkx = int(np.ceil(2.0*np.pi/(alength*kdist)))
    nky = int(np.ceil(2.0*np.pi/(blength*kdist)))
    nkz = int(np.ceil(2.0*np.pi/(clength*kdist)))
    nkxt = str(nkx)
    nkyt = str(nky)
    nkzt = str(nkz)
	
    # Print file(s)
    kpointfile = open('KPOINTS','w')
    kpointfile.write('k-density: %.1f\n'\
                     '0\n'\
                     'Gamma\n'\
                     '%s %s %s\n'
                     '0  0  0'%(kdensity,nkxt,nkyt,nkzt))
            
    return None

def launch_calc(location,nameofbatch):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Launch calculation using the bash script nameofbatch
    """
    bashsub = 'sbatch '+nameofbatch
    output = bash_commands_cwd(bashsub,wd = __root__+'/'+location+'/')
    
    print(output)
    
    return None

def launch_script(location,nameofbatch):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Launch calculation using the bash script nameofbatch
    """
    bashsub = './'+nameofbatch
    output = bash_commands_cwd(bashsub,wd = __root__+'/'+location+'/')
    
    print(output)
    
    return None

def relaunch_configs(unstable_files,filenames):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Relaunch calculations with unstable phonon modes
    
    Return: None
    """
    for i in range(len(unstable_files)):
        #Find what folder to look into.
        filenumber = unstable_files[i]
        
        #remove lines from data_extraction
        g = open('data_extraction','r')
        lines = g.readlines()
        g.close()
        
        #now rewrite file 
        g = open('data_extraction','w')
        for line in lines:
            if not line.startswith('free_energy_'):
                g.write(line)
        g.close()
                    
        #remove file from free_energies folder
        remove_file('free_energies/*')
        
        #remove old free_energy file
        remove_file('lattice_'+str(int(filenumber))+'/configs/outfile.free_energy')
        
        #remove folders for configurations 
        remove_file('-r lattice_'+str(int(filenumber))+'/configs/conf00*')
        
        #generate files
        generate_batch('TDEP2','batch.sh','')
                    
        #move files
        move_file('batch.sh','lattice_'+str(int(filenumber)),original=False)
        
        #launch calculation
        launch_calc('lattice_'+str(i),'batch.sh')
        
    sys.exit()    
    return None

def vasp_converge(calctype):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Simple function to determine if the VASP calculation completed 
    """
    check = "False"
    
    #stopping phrases in OUTCAR file
    Relaxation_phrase = ' reached required accuracy - stopping structural energy minimisation'
    Elastic_phrase    = ' Eigenvectors and eigenvalues of the dynamical matrix' 
    TDEP_phrase       = ' aborting loop because EDIFF is reached' 
    
    #now execute loop statements
    if calctype == 'Relaxation':
        with open('OUTCAR','r') as out:
              for line in out:
                    if Relaxation_phrase in line:
                        check = "True"
                        break
                    
    elif calctype == 'Elastic':
        with open('OUTCAR','r') as out:
              for line in out:
                    if Elastic_phrase in line:
                        check = "True"
                        break
                    
    elif calctype == 'TDEP':
        with open('OUTCAR','r') as out:
          for line in out:
                if TDEP_phrase in line:
                    check = "True"
                    break    
                
    print(check)
    
    return None

def check_computer():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Check supercomputer to determine if the correct files are found in 
             the correct directories for this program.  If they are not, the 
             program will print a warning (or build the files).
             
    Return: None
    """
    print('Checking directories for necessary files...')
    print(' ')
    print('Checking python version...')
    
    #check if python version is correct
    python_ver = sys.version_info
    cwd = __root__
    if python_ver.major >= 3:
        print('   The version of python is at least version 3.0')
    else:
        print('Please use python 3.0 or greater. Aborting')
        sys.exit()
        
    print('\n')
        
    print('Checking if necessary python modules exist...')
    #check if python required modules can be loaded
        
    try:
        import subprocess
        print('   SUBPROCESS could be imported.')
    except ImportError:
        print('ERROR: subpross not found. Please install this module using pip install --user subprocess')
        sys.exit()
        
    try:
        import numpy
        print('   NUMPY could be imported.')
    except ImportError:
        print('ERROR: numpy not found. Please install this module using pip install --user numpy')
        sys.exit()
        
    try:
        import scipy
        print('   SCIPY could be imported.')
    except ImportError:
        print('ERROR: scipy not found. Please install this module using pip install --user scipy')
        sys.exit()
        
    try:
        import linecache
        print('   LINECACHE could be imported.')
    except ImportError:
        print('ERROR: linecache not found. Please install this module using pip install --user linecache')
        sys.exit()
                      
    try: 
        import mendeleev 
        print('   MENDELEEV could be imported')
    except ImportError:
        print('ERROR: mendeleev not found. Please install this module using pip install --user mendeleev')
        sys.exit()
        
    #checking directory structure
    print(' ')
    print('Checking directory structure...')
    print('   ...checking for current working directory')
    print('      %s' %(cwd))
    if os.path.isdir(cwd):
        print('   Current working directory is found.')
    else:
        print('   Current working directory is not found.')
        print('   ERROR: Current working directory set as:')
        print('          %s' %cwd)
        print('          This is not the current directory.')
        sys.exit()
                
    #checking for pseudopotential files.
    print('   ...checking for pseudopotential files in')
    print('      %s' %(cwd+'/pseudos/'))
    if os.path.isdir(cwd+'/pseudos/'):
        print('   pseudos directory exists.')
    else:
        print('   pseudos directory not found.')
        print('   Creating directory now...')
        make_folder('pseudos')
        sys.exit()
        
    if os.path.exists(cwd+'/pseudos/Mg/POTCAR'): #check for Mg potcar file
        print('   pseudos directory has POTCAR files')
    else:
        print('   pseudos director is missing POTCAR files.')
        sys.exit()
        
    #check for TDEP programs
    print(' ')
    print('Checking for TDEP executables')
    #check for process_outcar
    if os.path.isdir(TDEPSR):
        print('   Directory for TDEP executables found.')
    else:
        print('   Directory to TDEP executables not found.')
        sys.exit()
        
    print('   looking for processing_outcar python file')
    if os.path.isfile(TDEPSR+'/process_outcar_5.3.py'):
        print('   process_outcar_5.3.py is found.')
    else:
        print('   process_outcar_5.3.py not found.')
        sys.exit()
        
    print('   looking for extract_forceconstants')
    if os.path.isfile(TDEPSR+'/extract_forceconstants'):
        print('   extract_forceconstants is found.')
    else:
        print('   extract_forceconstants not found.')
        sys.exit()
        
    print('   looking for phonon_dispersion_relations')
    if os.path.isfile(TDEPSR+'/phonon_dispersion_relations'):
        print('   phonon_dispersion_relations is found.')
    else:
        print('   phonon_dispersion_relations not found.')
        sys.exit()

    print('   looking for generate_structure')
    if os.path.isfile(TDEPSR+'/generate_structure'):
        print('   generate_structure is found.')
    else:
        print('   generate_structure not found.')
        sys.exit()       
        
    print('   looking for canonical_configuration')
    if os.path.isfile(TDEPSR+'/canonical_configuration'):
        print('   canonical_configuration is found.')
    else:
        print('   canonical_configuration not found.')
        sys.exit()          
                
    print(' ')
    print('All files and directories are found.  Happy computing!')    
    print(' ')
    
    return None

def get_lattice(file):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Read the lattice parameters from a poscar file and return the a, b, c, and volume
    
    Return: the three lattice parameters and volume of the cell
    """
    POSCAR_store = []
    #store poscar file for later use
    with open(file,'r') as POSCAR:
       for line in POSCAR:
           POSCAR_store = np.append(POSCAR_store,line)  
        
    #extract lengths of a,b and ,c lattice parameter
    scale = POSCAR_store[1].split() 
    avec  = POSCAR_store[2].split()
    bvec  = POSCAR_store[3].split()
    cvec  = POSCAR_store[4].split()
    for i in range(3):
        avec[i] = float(avec[i])*float(scale[0])
        bvec[i] = float(bvec[i])*float(scale[0])
        cvec[i] = float(cvec[i])*float(scale[0])
    a = np.linalg.norm(avec)
    b = np.linalg.norm(bvec)
    c = np.linalg.norm(cvec)
    vol = np.dot(cvec,np.cross(avec,bvec))
    
    return a,b,c,vol

def get_energy(file):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Read the total energy from the CONTCAR file and return the energy
    
    Return: Total energy
    """
    #read energy from...
    
    if file.endswith('OUTCAR'):
        with open(file,'r') as OUTCAR:
           for line in OUTCAR:
               if '  free  energy   TOTEN  =' in line:
                   l = line.split()
                   engtot = float(l[4])
               elif '  external pressure =' in line:
                   l = line.split()
                   presstot = float(l[3])
                   
    elif file.endswith('outfile.U0'):
        with open(file,'r') as U0:
            firstline = U0.readlines()
            engy      = firstline[0].split()
        engtot = float(engy[0])
        with open('../relaxation2/OUTCAR','r') as OUTCAR:
           for line in OUTCAR:
               if '  external pressure =' in line:
                   l = line.split()
                   presstot = float(l[3])
        
    return engtot,presstot

def read_POSCAR():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
   
    Purpose: Read the relaxed poscar file (Relaxation/CONTCAR) and extract the 
            lattice parameters.  From these parameters, determine how many poscar
            files need to be generated to determine the thermal expansion of the lattice.
            Lattice expansion is determined via v = (a*l/lo,b*l/lo,c*l/lo) such that l = lo*(1+percent)
            
    Return: number of different cells and an array of poscar names
    """
    diff_volumes = 0
    speccell     = 0
    poscarnames  = []
    percent      = 0.01
    #read in relaxed POSCAR file from relaxation calculation
    POSCAR_store = []
    
    diff_volumes,speccell = determine_volumes() 
        
    #store poscar file for later use
    with open('Relaxation/CONTCAR','r') as POSCAR:
       for line in POSCAR:
           POSCAR_store = np.append(POSCAR_store,line)  
        
    #extract lengths of a,b and ,c lattice parameter
    avec = POSCAR_store[2].split()
    bvec = POSCAR_store[3].split()
    cvec = POSCAR_store[4].split()
    a = np.linalg.norm(avec)
    b = np.linalg.norm(bvec)
    c = np.linalg.norm(cvec)
     
    #now that we know how many volumes to use, we can build new poscar files
    if diff_volumes == 6 and speccell == 0:
       for i in range(diff_volumes):
           an = a*(1.0+percent*(i-1.0))
           bn = b*(1.0+percent*(i-1.0))
           cn = c*(1.0+percent*(i-1.0))
           
           anew = [np.float(avec[0])*an/a,np.float(avec[1])*an/a,np.float(avec[2])*an/a] 
           bnew = [np.float(bvec[0])*bn/b,np.float(bvec[1])*bn/b,np.float(bvec[2])*bn/b]
           cnew = [np.float(cvec[0])*cn/c,np.float(cvec[1])*cn/c,np.float(cvec[2])*cn/c] 
             
           #change lines in POSCAR
           POSCAR_store[2] = str(anew[0])+' '+str(anew[1])+' '+str(anew[2])+'\n'
           POSCAR_store[3] = str(bnew[0])+' '+str(bnew[1])+' '+str(bnew[2])+'\n'
           POSCAR_store[4] = str(cnew[0])+' '+str(cnew[1])+' '+str(cnew[2])+'\n'
           
           with open('POSCAR_'+str(i),'w') as f:
               for j in range(len(POSCAR_store)):
                   f.write(POSCAR_store[j])
           poscarnames = np.append(poscarnames,'POSCAR_'+str(i))
               
    elif diff_volumes == 36:
       cc = 0
      
       if  speccell == 0:
           numb = int(np.sqrt(diff_volumes))
           for i in range(numb):
               for j in range(numb):
                   an = a*(1.0+percent*(i-1.0))        
                   bn = b*(1.0+percent*(i-1.0)) 
                   cn = c*(1.0+percent*(j-1.0)) 
                   
                   anew = [np.float(avec[0])*an/a,np.float(avec[1])*an/a,np.float(avec[2])*an/a] 
                   bnew = [np.float(bvec[0])*bn/b,np.float(bvec[1])*bn/b,np.float(bvec[2])*bn/b]
                   cnew = [np.float(cvec[0])*cn/c,np.float(cvec[1])*cn/c,np.float(cvec[2])*cn/c]                   
                   
                   #change lines in POSCAR
                   POSCAR_store[2] = str(anew[0])+' '+str(anew[1])+' '+str(anew[2])+'\n'
                   POSCAR_store[3] = str(bnew[0])+' '+str(bnew[1])+' '+str(bnew[2])+'\n'
                   POSCAR_store[4] = str(cnew[0])+' '+str(cnew[1])+' '+str(cnew[2])+'\n'
                                      
                   with open('POSCAR_'+str(cc),'w') as f:
                       for j in range(len(POSCAR_store)):
                           f.write(POSCAR_store[j])
                   poscarnames = np.append(poscarnames,'POSCAR_'+str(cc))
                   cc +=1
                   
       elif  speccell == 1:
           numb = int(np.sqrt(diff_volumes))
           for i in range(numb):
               for j in range(numb):
                   an = a*(1.0+percent*(i-1.0))        
                   bn = b*(1.0+percent*(j-1.0)) 
                   cn = c*(1.0+percent*(i-1.0)) 
                   
                   anew = [np.float(avec[0])*an/a,np.float(avec[1])*an/a,np.float(avec[2])*an/a] 
                   bnew = [np.float(bvec[0])*bn/b,np.float(bvec[1])*bn/b,np.float(bvec[2])*bn/b]
                   cnew = [np.float(cvec[0])*cn/c,np.float(cvec[1])*cn/c,np.float(cvec[2])*cn/c]                   
                   
                   #change lines in POSCAR
                   POSCAR_store[2] = str(anew[0])+' '+str(anew[1])+' '+str(anew[2])+'\n'
                   POSCAR_store[3] = str(bnew[0])+' '+str(bnew[1])+' '+str(bnew[2])+'\n'
                   POSCAR_store[4] = str(cnew[0])+' '+str(cnew[1])+' '+str(cnew[2])+'\n'
                   
                   with open('POSCAR_'+str(cc),'w') as f:
                       for j in range(len(POSCAR_store)):
                           f.write(POSCAR_store[j])                   
                   poscarnames = np.append(poscarnames,'POSCAR_'+str(cc))
                   cc+=1
                   
       elif  speccell == 2:
           numb = int(np.sqrt(diff_volumes))
           for i in range(numb):
               for j in range(numb):
                   an = a*(1.0+percent*(i-1.0))        
                   bn = b*(1.0+percent*(j-1.0)) 
                   cn = c*(1.0+percent*(j-1.0))
                   
                   anew = [np.float(avec[0])*an/a,np.float(avec[1])*an/a,np.float(avec[2])*an/a] 
                   bnew = [np.float(bvec[0])*bn/b,np.float(bvec[1])*bn/b,np.float(bvec[2])*bn/b]
                   cnew = [np.float(cvec[0])*cn/c,np.float(cvec[1])*cn/c,np.float(cvec[2])*cn/c]                   
                   
                   #change lines in POSCAR
                   POSCAR_store[2] = str(anew[0])+' '+str(anew[1])+' '+str(anew[2])+'\n'
                   POSCAR_store[3] = str(bnew[0])+' '+str(bnew[1])+' '+str(bnew[2])+'\n'
                   POSCAR_store[4] = str(cnew[0])+' '+str(cnew[1])+' '+str(cnew[2])+'\n'

                   with open('POSCAR_'+str(cc),'w') as f:
                       for j in range(len(POSCAR_store)):
                           f.write(POSCAR_store[j])
                   poscarnames = np.append(poscarnames,'POSCAR_'+str(cc))
                   cc +=1
                           
    elif diff_volumes == 216 and speccell == 0:
       cc = 0
       numb = int(diff_volumes**(1.0/3.0))
       for i in range(numb):
           for j in range(numb):
               for k in range(numb):
                   an = a*(1.0+percent*(i-1.0))        
                   bn = b*(1.0+percent*(j-1.0)) 
                   cn = c*(1.0+percent*(k-1.0))    
                   
                   anew = [np.float(avec[0])*an/a,np.float(avec[1])*an/a,np.float(avec[2])*an/a] 
                   bnew = [np.float(bvec[0])*bn/b,np.float(bvec[1])*bn/b,np.float(bvec[2])*bn/b]
                   cnew = [np.float(cvec[0])*cn/c,np.float(cvec[1])*cn/c,np.float(cvec[2])*cn/c]                   
                   
                   #change lines in POSCAR
                   POSCAR_store[2] = str(anew[0])+' '+str(anew[1])+' '+str(anew[2])+'\n'
                   POSCAR_store[3] = str(bnew[0])+' '+str(bnew[1])+' '+str(bnew[2])+'\n'
                   POSCAR_store[4] = str(cnew[0])+' '+str(cnew[1])+' '+str(cnew[2])+'\n'
        
                   with open('POSCAR_'+str(cc),'w') as f:
                       for m in range(len(POSCAR_store)):
                           f.write(POSCAR_store[m])
                   poscarnames = np.append(poscarnames,'POSCAR_'+str(cc))        
                   cc +=1
                      
    return diff_volumes,poscarnames

def determine_volumes():
    """
    Author: Nicholas Pike 
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determines the number of volumes by reading the poscar file.
    
    Return: number of volumes and the cell identification number
    """
    POSCAR_store = []
    with open('Relaxation/CONTCAR','r') as POSCAR:
       for line in POSCAR:
           POSCAR_store = np.append(POSCAR_store,line)  
        
    #extract lengths of a,b and ,c lattice parameter
    avec = POSCAR_store[2].split()
    bvec = POSCAR_store[3].split()
    cvec = POSCAR_store[4].split()
    
    #find the magnitude of the lattice vectors
    a = np.linalg.norm(avec)
    b = np.linalg.norm(bvec)
    c = np.linalg.norm(cvec)
            
    if ( np.abs(a-b)<=difftol and np.abs(b -c) <= difftol 
        and np.abs(a-c)<=difftol  ):
       #isotropic. 6 different volumes, ax, by, and cz are the only non-zero lattice parameters
       diff_volumes = 6
       speccell = 0
       
    elif ( np.abs(a-b)<= difftol and np.abs(a-c) > difftol and np.abs(b-c) > difftol ):
       #a and b are the same, c is different, 36 volumes 
       diff_volumes = 36
       speccell = 0
       
    elif ( np.abs(a-b)> difftol and np.abs(a-c) <= difftol and np.abs(b-c) > difftol ):
       #a and c are the same, b is different, 36 volumes 
       diff_volumes = 36
       speccell = 1
       
    elif ( np.abs(a-b)> difftol and np.abs(b -c)<= difftol and np.abs(a-c) > difftol ):
       #b and c are the same, a is different, 36 volumes , ax, by, and cz are the only non-zero lattice parameters
       diff_volumes = 36
       speccell = 2
       
    elif a != b and b != c:
       #all vectors are different, 216 volumes and , ax, by, and cz are the only non-zero lattice parameters
       diff_volumes = 216
       speccell = 0
       
    else:
       print('Something bad happened, or differnet symmetry read in, when reading the POSCAR file')
       sys.exit()
       
    return diff_volumes,speccell

def find_gcd(x, y):
    """
    Determines the gcd for an array of values.
    """
    while(y):
        x, y = y, x % y
    return x

def generate_LOTO():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: This file extracts the dielectric tensor and born effective charge tensors from data_extract
             This code is very similiar to a code written by Olle Hellman that does the same thing
    """
    #get data from data_extraction
    dietensor = np.zeros(shape=(3,3))
    bectensor = [] 
    with open('data_extraction','r') as datafile:
        for i,line in enumerate(datafile):
            if 'dielectric' in line:
                l = line.split()
                dietensor[0][0]= float(l[1])
                dietensor[0][1]= float(l[2])
                dietensor[0][2]= float(l[3])
                dietensor[1][0]= float(l[4])
                dietensor[1][1]= float(l[5])
                dietensor[1][2]= float(l[6])
                dietensor[2][0]= float(l[7])
                dietensor[2][1]= float(l[8])
                dietensor[2][2]= float(l[9])
                
            elif 'natom' in line:
                l = line.split()
                natom = l[1]
            elif 'bectensor' in line:
                bectensor = np.zeros(shape=(int(natom),3,3))
                for j in range(int(natom)):
                    if j == 0:
                        data1 = linecache.getline('data_extraction',i+j+1).split()
                        bectensor[j][0][0] = float(data1[2])
                        bectensor[j][0][1] = float(data1[3])
                        bectensor[j][0][2] = float(data1[4])
                        bectensor[j][1][0] = float(data1[5])
                        bectensor[j][1][1] = float(data1[6])
                        bectensor[j][1][2] = float(data1[7])
                        bectensor[j][2][0] = float(data1[8])
                        bectensor[j][2][1] = float(data1[9])
                        bectensor[j][2][2] = float(data1[10]) 
                    else:
                        data1 = linecache.getline('data_extraction',i+j+1).split()
                        bectensor[j][0][0] = float(data1[1])
                        bectensor[j][0][1] = float(data1[2])
                        bectensor[j][0][2] = float(data1[3])
                        bectensor[j][1][0] = float(data1[4])
                        bectensor[j][1][1] = float(data1[5])
                        bectensor[j][1][2] = float(data1[6])
                        bectensor[j][2][0] = float(data1[7])
                        bectensor[j][2][1] = float(data1[8])
                        bectensor[j][2][2] = float(data1[9]) 
    #now print data to file
    f = open('infile.lotosplitting','w')
    f.write('%f %f %f\n' %(dietensor[0][0],dietensor[0][1],dietensor[0][2]))
    f.write('%f %f %f\n' %(dietensor[1][0],dietensor[1][1],dietensor[1][2]))
    f.write('%f %f %f\n' %(dietensor[2][0],dietensor[2][1],dietensor[2][2]))
    for i in range(int(natom)):
        f.write('%f %f %f\n' %(bectensor[i][0][0],bectensor[i][0][1],bectensor[i][0][2]))
        f.write('%f %f %f\n' %(bectensor[i][1][0],bectensor[i][1][1],bectensor[i][1][2]))
        f.write('%f %f %f\n' %(bectensor[i][2][0],bectensor[i][2][1],bectensor[i][2][2]))
    f.close()
    
    return None

def find_debye():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose:Determine the debye temperature from the calculated elastic tensor
    
    Return: Debye Temperature
    """
    debye_temp = 0
    density    = 0
    masstot    = 0
    molarmass  = 0
    data       = ['',0,0,0,0]
    molar      = []
    eltensor   = np.zeros(shape=(6,6))
    
    #read information from data_extraction file
    with open('data_extraction','r') as datafile:
        for line in datafile:
            if line.startswith('elastic'):
                l = line.split()
                eltensor[0][0]= float(l[1])
                eltensor[0][1]= float(l[2])
                eltensor[0][2]= float(l[3])
                eltensor[0][3]= float(l[4])
                eltensor[0][4]= float(l[5])
                eltensor[0][5]= float(l[6])
                eltensor[1][0]= float(l[7])
                eltensor[1][1]= float(l[8])
                eltensor[1][2]= float(l[9])
                eltensor[1][3]= float(l[10])
                eltensor[1][4]= float(l[11])
                eltensor[1][5]= float(l[12])
                eltensor[2][0]= float(l[13])
                eltensor[2][1]= float(l[14])
                eltensor[2][2]= float(l[15])
                eltensor[2][3]= float(l[16])
                eltensor[2][4]= float(l[17])
                eltensor[2][5]= float(l[18])
                eltensor[3][0]= float(l[19])
                eltensor[3][1]= float(l[20])
                eltensor[3][2]= float(l[21])
                eltensor[3][3]= float(l[22])
                eltensor[3][4]= float(l[23])
                eltensor[3][5]= float(l[24])
                eltensor[4][0]= float(l[25])
                eltensor[4][1]= float(l[26])
                eltensor[4][2]= float(l[27])
                eltensor[4][3]= float(l[28])
                eltensor[4][4]= float(l[29])
                eltensor[4][5]= float(l[30])
                eltensor[5][0]= float(l[32])
                eltensor[5][1]= float(l[32])
                eltensor[5][2]= float(l[33])
                eltensor[5][3]= float(l[34])
                eltensor[5][4]= float(l[35])
                eltensor[5][5]= float(l[36])
                data[1] = eltensor
                
            elif line.startswith('volume'):
                l = line.split()
                data[3] = float(l[1])*ang_to_m**3
                
            elif line.startswith('atpos'):
                l = line.split()
                u,data[4] = np.unique(l[1:],return_counts = True)
                for i in range(len(u)):
                    m  = element(u[i]).mass
                    molar = np.append(molar,m)
                data[2] = molar
                    
            elif line.startswith('natom'):
                l = line.split()
                natom = float(l[1])
                   
    #calculate molar mass
    for i in range(len(data[2])):
        molarmass += data[2][i]
    
    #calculate density
    for i in range(len(data[2])):
        masstot +=data[2][i]*data[4][i] #data[2] is the mass data[4] is the multiplicity
    density = masstot*amu_kg/data[3]/kgm3_to_cm3 #data[3] is the volume in m**3
    
    #number of unit cells 
    uc_count = 0
    for i in range(len(data[4])):
        uc_count = find_gcd(uc_count,data[4][i])
    
    
    #calculate number of atoms per molecule
    atom_molecule = natom/uc_count
    
    #calculate compliance tensor
    if np.all(data[1]==0):
        s = np.zeros(shape=(6,6))
    else:
        s = np.linalg.inv(data[1])
    
    #calculate bulk and shear modulus
    B  = 1.0/9.0*((data[1][0][0]+data[1][1][1]+data[1][2][2])+2.0*(data[1][0][1]+data[1][0][2]+data[1][1][2]))
    BR = 1.0/((s[0,0]+s[1,1]+s[2,2])+2.0*(s[0,1]+s[1,2]+s[0,2]))
    G  = 1.0/15.0*((data[1][0][0]+data[1][1][1]+data[1][2][2])-(data[1][0][1]+data[1][0][2]+data[1][1][2])+3.0*(data[1][3][3]+data[1][4][4]+data[1][5][5]))
    GR = 15.0/(4.0*(s[0,0]+s[1,1]+s[3,3])-4.0*(s[0,1]+s[1,2]-s[0,2])+3.0*(s[3,3]+s[4,4]+s[5,5])) 

    #calculate universal elastic anisotropy constant
    Au = 5.0*(G/GR)+B/BR - 6.0
    
    #calculate shear velocity
    vs = np.sqrt(G*kbar_to_GPa*1E9/(density*kgm3_to_cm3))
    
    #calculate longitudional velocity
    vl  = np.sqrt((B+4.0/3.0*G)*kbar_to_GPa*1E9/(density*kgm3_to_cm3))
    
    #acoustic velocity
    va = (1.0/3.0*((1.0/vl**3)+(2.0/vs**3)))**(-1.0/3.0)
    
    #Calculate Debye temperature
    debye_temp  = h/kb*((3.0*atom_molecule*Na*density)/(4.0*np.pi*molarmass))**(1.0/3.0)*va*m_to_cm
    
    #print information to file
    printdebye = True
    with open('data_extraction','r') as f:
        for line in f:
            if 'Debye' in line:
                printdebye = False
                
    if printdebye == True:
        f1 =  open('data_extraction','a')
        f1.write('Debye %f\n'%debye_temp)
        f1.write('Bulk Mod %f\n' %B)
        f1.write('# Additional elastic information\n')
        f1.write('#Bulk Mod (Voigt Avg): %.2f kbar\n' %B)
        f1.write('#Bulk Mod (Reuss Avg): %.2f kbar\n' %BR)
        f1.write('#Shear Mod (Voigt Avg): %.2f kbar\n' %G)
        f1.write('#Shear Mod (Reuss Avg): %.2f kbar\n' %GR)
        f1.write('#Universal Elastic Aniostropy: %.2f\n' %Au)
        f1.write('#Avg. Vel.  %.2f m/s\n'%va)
        f1.write('#Shear Vel. %.2f m/s\n'%vs)
        f1.write('#Long. Vel. %.2f m/s\n'%vl)
        f1.close()
            
    return debye_temp


def sortflags():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determine what to calculate based on the input flags
    """
    
    if len(sys.argv) == 1:
        #check if the user did not enter anything
        print('Use python ACTE.py --help to view the help menu and \nto learn how to run the program.')
        sys.exit()
    else:
        i = 1
        while i < len(sys.argv):
            tagfound = False
            if sys.argv[i] == '--help':
                tagfound = True
                print('\n\n')
                print('--help\t\t Prints this help menu.')
                print('--usage\t\t Gives an example of how to use this program.')
                print('--author\t Gives author information.')
                print('--version\t Gives the version of this program.')
                print('--email\t\t Provides an email address of the primary author.')
                print('--bug\t\t Provides an email address for bug reports.')
                print('\nThe following commands are the main executables of the program.')
                print('-----------------------------------------------------------------\n')
                print('--check_sys\t    Checks directory for files and executables needed to run this script.')
                print('--relaxation \t    Launches relaxation, dielectric, and elastic calculations.')
                print('--build_cells \t    Determines the number of volumes needed for lattice calculations.')
                print('--relaunch_tdep     Relaunches the tdep calculations.')
                print('--thermal_expansion Calculates the thermal expansion of the material.')
                print('\nThe following tags are used internally by various scripts.')
                print('-----------------------------------------------------------------\n')
                print('--move_file \t Moves the selected file to a new folder.')
                print('--copy_file \t Copies the selected file to a new location.')
                print('--generate_batch Generates batch submission script.')
                print('--make_KPOINTS \t Generates a k-point file with a determined density.')
                print('--launch_calc \t Launches parallel calculation.')
                print('--outcar\t Gather information from outcar file.')
                print('--vasp_converge  Checks the convergence of the calculation.')
                i+=1
                sys.exit()
            
            elif sys.argv[i] == '--usage':
                tagfound = True
                print('--usage\t To use this program use the following in the command line.\n python ACTE.py --TAG data_extraction_file')
                i+=1
                sys.exit()
        
            elif sys.argv[i] == '--author':
                tagfound = True
                print('--author\t The author of this program is %s' %__author__)
                i+=1
                sys.exit()
        
            elif sys.argv[i] == '--version':
                tagfound = True
                print('--version\t This version of the program is %s \n\n  This version allows the user'
                      ' to read in information gathered with a\n different python script and calculate\n '
                      ' the pyroelectric coefficients from a first principles calculation.' %__version__)
                i+=1
                sys.exit()
    
            elif sys.argv[i] == '--email':
                tagfound = True
                print('--email\t Please send questions and comments to %s' %__email__)
                i+=1
                sys.exit()
        
            elif sys.argv[i] == '--bug':
                tagfound = True
                print('--bug \t Please send bug reports to %s.  Make sure you include as much information as\n as possible about your calculation.' %__email__)
                print('\t\t It is suggested that you include the script, POSCAR file, and any information you think is useful.')
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--check_sys':
                tagfound = True
                check_computer()
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--relaxation':
                tagfound = True
                #In parent directory, create folders for Relaxation, Dielectric Tensor, and Elastic Tensor calculations
                make_folder('Relaxation')
                make_folder('Elastic')
                
                #Create input files for VASP calculations and store in the correct folders
                generate_batch('Relaxation','batch.sh','')
                move_file('batch.sh','Relaxation',original = False)
                
                generate_INCAR('Relaxation')
                move_file('INCAR','Relaxation',original = False)
                generate_INCAR('Elastic')
                move_file('INCAR','Elastic',original = False)
                
                generate_KPOINT(kdensity)
                move_file('KPOINTS','Relaxation',original = True)
                generate_KPOINT(int(2.0*kdensity))  #double kdensity for better elastic constants
                move_file('KPOINTS','Elastic',original = True)
        
                generate_POTCAR()
                move_file('POTCAR','Relaxation',original = True)
                move_file('POTCAR','Elastic',original = True)
                
                #POSCAR moved to relaxation first
                move_file('POSCAR','Relaxation',original = True)
                
                #submit relaxation calculation
                launch_calc('Relaxation','batch.sh')
                i+=1
                sys.exit()
            
            elif sys.argv[i] == '--build_cells':
                tagfound = True
                #read in the poscar to determine how many files to generate
                diff_volumes,poscarnames = read_POSCAR() 
                                                                
                #determine debye temperature
                find_debye()
                
                #generate lotosplitting file
                generate_LOTO()
                
                #generate Incar and batch files
                generate_INCAR('TDEP_rlx')
                copy_file('INCAR','INCAR_rlx')
                generate_INCAR('TDEP')
                generate_batch('TDEP','batch.sh','')
                generate_KPOINT(kdensity)
                        
                for j in range(diff_volumes):
                    make_folder('lattice_'+str(j))
                    move_file('batch.sh','lattice_'+str(j),original=True)
                    move_file('INCAR','lattice_'+str(j),original=True) 
                    move_file('INCAR_rlx','lattice_'+str(j),original=True)
                    move_file('POTCAR','lattice_'+str(j),original=True)
                    move_file('KPOINTS','lattice_'+str(j),original=True)
                    move_file('POSCAR_'+str(j),'lattice_'+str(j),original=False)
                    launch_calc('lattice_'+str(j),'batch.sh')
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--relaunch_tdep':
                tagfound = True
                diff_volumes, speccell = determine_volumes() 
                for ii in range(diff_volumes):
                    launch_calc('lattice_'+str(ii),'batch.sh')
                i+=1
                sys.exit()
           
            elif sys.argv[i] == '--thermal_expansion':
                tagfound = True
                #set tags as false to start
                withbounds = False
                withsolver = False
                withBEC    = False
                withpoly   = [False,0.1,2,False]    #requires more than a truth value, also low and high multipliers
                store_data = []
                ii = i+1
                while ii <len(sys.argv):
                    if sys.argv[ii] =='--bounds':
                        withbounds = True
                        ii += 1
                    elif sys.argv[ii] == '--solver':
                        withsolver = True
                        ii += 1
                    elif sys.argv[ii] == '--BEC':
                        withBEC   = True
                        ii += 1
                    elif sys.argv[ii] == '--polyfit':
                        withpoly[0] = True
                        try:
                            withpoly[1] = sys.argv[ii+1]
                            withpoly[2] = sys.argv[ii+2]
                            try:
                                withpoly[3] = sys.argv[ii+3]
                                ii += 1
                            except:
                                pass   
                            ii +=3

                        except:
                            #use default values instead
                            withpoly[1] = 0.1
                            withpoly[2] = 2
                            withpoly[3] = False
                            ii +=2
                    else:
                        print('%s is not a valid tag.  See help manual.' %sys.argv[ii])
                        sys.exit()
                tags = [withbounds,withsolver,withBEC,withpoly]
                
                #looks for the file we already named
                DFT_INPUT = 'data_extraction'
                    
                diff_volumes,speccell = determine_volumes()
                
                if not os.path.isdir('free_energies'):
                    make_folder('free_energies')
                
                #check data_extraction for the correct header
                printline        = True
                calc_free_energy = True
                with open('data_extraction','r') as f:
                    for line in f:
                        if 'Free energy on grid filenames' in line:
                            printline = False
                        elif 'free_energy_' in line:
                           calc_free_energy = False #checks the case the header was written by the free energy files are not produced                   
                           
                if printline == True:
                    with open('data_extraction','a') as f:
                        f.write('Free energy on grid filenames\n')
                        calc_free_energy = True
                        
                if calc_free_energy == True:
                    #determine what to do for different symmetries if not already done  
                    print('Starting calculation of dispersion relations and density of states.')             
                    if diff_volumes == 6:
                        store_data = np.zeros((diff_volumes,7))
                        print('   Looking for %s different volume files...' %diff_volumes)
                        for i in range(diff_volumes):
                            writeline = False
                            if os.path.isfile('lattice_'+str(i)+'/configs/outfile.free_energy'):
                                print('   Free_energy file found for volume %s' %i)
                                #gather all files into a single place by copying the file from its 
                                #current folder to a new destination and rename it
                                a,b,c,vol = get_lattice('lattice_'+str(i)+'/POSCAR')

                                engtot,pressure = get_energy('lattice_'+str(i)+'/relaxation2/OUTCAR')
                                
                                #store thermo data
                                store_data[i][0] = i
                                store_data[i][1] = a
                                store_data[i][2] = b
                                store_data[i][3] = c
                                store_data[i][4] = vol
                                store_data[i][5] = engtot
                                store_data[i][6] = pressure
                                
                                newfilename = 'free_energy_'+str(i)+'_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(engtot)+'_'+str(vol)
                                copy_file('lattice_'+str(i)+'/configs/outfile.free_energy','free_energies/'+newfilename)
                                with open('data_extraction','r') as f:
                                    for line in f:
                                        if not newfilename in line:
                                            writeline = True
                                if writeline == True:
                                    with open('data_extraction','a') as f:
                                        f.write(newfilename+'\n')
                            else:
                                #go into the file folder for each volume calculation and run tdep
                                if withBEC == True or withsolver == True:
                                    generate_batch('script','lattice_'+str(i)+'/configs/gather.sh',tags)
                                else:
                                    generate_batch('script','lattice_'+str(i)+'/configs/gather.sh','')
                                    
                                launch_script('lattice_'+str(i)+'/configs','gather.sh')
                                
                                #gather all files into a single place by copying the file from its 
                                #current folder to a new destination and rename it
                                a,b,c,vol = get_lattice('lattice_'+str(i)+'/POSCAR')

                                engtot,pressure = get_energy('lattice_'+str(i)+'/relaxation2/OUTCAR')
                                
                                #store thermo data
                                store_data[i][0] = i
                                store_data[i][1] = a
                                store_data[i][2] = b
                                store_data[i][3] = c
                                store_data[i][4] = vol
                                store_data[i][5] = engtot
                                store_data[i][6] = pressure
                                
                                newfilename = 'free_energy_'+str(i)+'_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(engtot)+'_'+str(vol)
                                copy_file('lattice_'+str(i)+'/configs/outfile.free_energy','free_energies/'+newfilename)
                                with open('data_extraction','a') as f:
                                    f.write(newfilename+'\n')
                            
                    elif diff_volumes == 36:
                        store_data = np.zeros((diff_volumes,7))
                        print('   Looking for %s different volume files...' %diff_volumes)
                        cc= 0
                        for i in range(int(np.sqrt(diff_volumes))):
                            for j in range(int(np.sqrt(diff_volumes))):
                                writeline = False
                                if os.path.isfile('lattice_'+str(cc)+'/configs/outfile.free_energy'):
                                    print('   Free_energy file found for volume %s' %cc)
                                    #gather all files into a single place by copying the file from its 
                                    #current folder to a new destination and rename it
                                    a,b,c,vol = get_lattice('lattice_'+str(cc)+'/POSCAR')

                                    engtot,pressure = get_energy('lattice_'+str(cc)+'/relaxation2/OUTCAR')
                                    
                                    #store thermo data
                                    store_data[cc][0] = cc
                                    store_data[cc][1] = a
                                    store_data[cc][2] = b
                                    store_data[cc][3] = c
                                    store_data[cc][4] = vol
                                    store_data[cc][5] = engtot
                                    store_data[cc][6] = pressure
                                
                                    newfilename = 'free_energy_'+str(i)+str(j)+'_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(engtot)+'_'+str(vol)
                                    copy_file('lattice_'+str(cc)+'/configs/outfile.free_energy','free_energies/'+newfilename)
                                    cc+=1
                                    with open('data_extraction','r') as f:
                                        for line in f:
                                            if not newfilename in line:
                                                writeline = True
                                    if writeline == True:
                                        with open('data_extraction','a') as f:
                                            f.write(newfilename+'\n')
                                else:
                                    #go into the file folder for each volume calculation and run tdep
                                    if withBEC == True or withsolver == True:
                                        generate_batch('script','lattice_'+str(cc)+'/configs/gather.sh',tags)
                                    else:
                                        generate_batch('script','lattice_'+str(cc)+'/configs/gather.sh','')
                                        
                                    launch_script('lattice_'+str(cc)+'/configs','gather.sh')
                                    
                                    #gather all files into a single place by copying the file from its 
                                    #current folder to a new destination and rename it
                                    a,b,c,vol = get_lattice('lattice_'+str(cc)+'/POSCAR')

                                    engtot,pressure = get_energy('lattice_'+str(cc)+'/relaxation2/OUTCAR')
                                    
                                    #store thermo data
                                    store_data[cc][0] = cc
                                    store_data[cc][1] = a
                                    store_data[cc][2] = b
                                    store_data[cc][3] = c
                                    store_data[cc][4] = vol
                                    store_data[cc][5] = engtot
                                    store_data[cc][6] = pressure
                                    
                                    newfilename = 'free_energy_'+str(i)+str(j)+'_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(engtot)+'_'+str(vol)
                                    copy_file('lattice_'+str(cc)+'/configs/outfile.free_energy','free_energies/'+newfilename)
                                    cc+=1
                                    with open('data_extraction','a') as f:
                                        f.write(newfilename+'\n')
                                
                    elif diff_volumes == 216:
                        store_data = np.zeros((diff_volumes,7))
                        print('   Looking for %s different volume files...' %diff_volumes)
                        cc= 0
                        for i in range(int(diff_volumes**(1/3))):
                            for j in range(int(diff_volumes**(1/3))):
                                for k in range(int(diff_volumes**(1/3))):
                                    writeline = False
                                    if os.path.isfile('lattice_'+str(cc)+'/configs/outfile.free_energy'):
                                        print('   Free_energy file found for volume %s' %cc)
                                        #gather all files into a single place by copying the file from its 
                                        #current folder to a new destination and rename it
                                        a,b,c,vol = get_lattice('lattice_'+str(cc)+'/POSCAR')

                                        engtot,pressure = get_energy('lattice_'+str(cc)+'/relaxation2/OUTCAR')
                                        
                                        #store thermo data
                                        store_data[cc][0] = cc
                                        store_data[cc][1] = a
                                        store_data[cc][2] = b
                                        store_data[cc][3] = c
                                        store_data[cc][4] = vol
                                        store_data[cc][5] = engtot
                                        store_data[cc][6] = pressure
                                
                                        newfilename = 'free_energy_'+str(i)+str(j)+str(k)+'_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(engtot)+'_'+str(vol)
                                        copy_file('lattice_'+str(cc)+'/configs/outfile.free_energy','free_energies/'+newfilename)
                                        cc+=1
                                        with open('data_extraction','r') as f:
                                            for line in f:
                                                if not newfilename in line:
                                                    writeline = True
                                        if writeline == True:
                                            with open('data_extraction','a') as f:
                                                f.write(newfilename+'\n')
                                    else:
                                        #go into the file folder for each volume calculation and run tdep
                                        if withBEC == True or withsolver == True:
                                            generate_batch('script','lattice_'+str(cc)+'/configs/gather.sh',tags)
                                        else:
                                            generate_batch('script','lattice_'+str(cc)+'/configs/gather.sh','')
                                            
                                        launch_script('lattice_'+str(cc)+'/configs','gather.sh')
                                        
                                        #gather all files into a single place by copying the file from its 
                                        #current folder to a new destination and rename it
                                        a,b,c,vol = get_lattice('lattice_'+str(cc)+'/POSCAR')

                                        engtot,pressure = get_energy('lattice_'+str(cc)+'/relaxation2/OUTCAR')
                                        
                                        #store thermo data
                                        store_data[cc][0] = cc
                                        store_data[cc][1] = a
                                        store_data[cc][2] = b
                                        store_data[cc][3] = c
                                        store_data[cc][4] = vol
                                        store_data[cc][5] = engtot
                                        store_data[cc][6] = pressure
                                        
                                        newfilename = 'free_energy_'+str(i)+str(j)+str(k)+'_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(engtot)+'_'+str(vol)
                                        copy_file('lattice_'+str(cc)+'/configs/outfile.free_energy','free_energies/'+newfilename)
                                        cc+=1
                                        with open('data_extraction','a') as f:
                                            f.write(newfilename+'\n')

                #print thermo data to file
                with open('data_extraction','a') as f: 
                    f.write('#Thermodynamics data\n# Vol # \t a \t b \t c \t vol \t Energy\\atom \t Pressure\n')                                           
                    for i in range(len(store_data)):
                         f.write('%i %f %f %f %f %f %f\n' 
                                 %(store_data[i][0],store_data[i][1],store_data[i][2],
                                   store_data[i][3],store_data[i][4],store_data[i][5],store_data[i][6]))
                
                print('Starting calculation of temperature-dependent properties.')
                #run main thermal program to extract all temperature dependent quantities
                main_thermal(DFT_INPUT,tags)
                i+=1
                sys.exit()
    
            elif sys.argv[i] == '--move_file':
                tagfound = True
                move_file(sys.argv[i+1],sys.argv[i+2],sys.argv[i+3])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--copy_file':
                tagfound = True
                copy_file(sys.argv[i+1],sys.argv[i+2])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--launch_calc':
                tagfound = True
                launch_calc(sys.argv[i+1],sys.argv[i+2])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--generate_batch':
                tagfound = True
                generate_batch(sys.argv[i+1],sys.argv[i+2],'')
                i+=1
                sys.exit()
            
            elif sys.argv[i] == '--make_KPOINTS':
                tagfound = True
                generate_KPOINT(float(sys.argv[i+1]))
                i+=1
                sys.exit()
            
            elif sys.argv[i] == '--make_folder':
                tagfound = True
                make_folder(float(sys.argv[i+1]))
                i+=1
                sys.exit()
                            
            elif sys.argv[i] == '--vasp_converge':
                tagfound = True
                vasp_converge(sys.argv[i+1])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--outcar':
                tagfound = True
                gather_outcar()
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--debye':
                tagfound = True
                find_debye()
                i+=1
                sys.exit()
                
            if tagfound != True:
                print('ERROR: The tag %s is not reconized by the program.  Please use --help \n to see available tags.' %sys.argv[i])
                sys.exit()
        i+= 1        

    return None

"""
Run main program for windows machine, others will work automatically
"""       

#starts main program for windows machines... has no effect for other machine types
if __name__ == '__main__':
    __author__     = 'Nicholas Pike'
    __copyright__  = 'none'
    __credits__    = 'none'
    __license__    = 'none'
    __version__    = '0.0'
    __maintainer__ = 'Nicholas Pike'
    __email__      = 'Nicholas.pike@sintef.no'
    __status__     = 'experimental'
    __date__       = 'October 2018'
       
    #determine what type of calculation to run via flags and tags
    sortflags()
    
"""
End program

Happy Calculating!
"""