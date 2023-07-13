#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from .grid import *
from .pv import *
from .omega import *
from .inout import *
from .fftlib import *


def comp_vel(psi, Delta, bc=None, loc='center'):

  '''
  Compute velocity at cell center or cell faces

  u = -d psi /dy
  v =  d psi /dx

  If psi is defined at cell center then
  **warning**
  cell faces do not correspond to a C-grid:
  v_f is defined at the *eas-west* faces
  u_f is defined at the *north-south* faces

  Cell center vs faces:

  +----u_f----+----u_f----+
  |           |           |
  |           |           |
 v_f  u,v_c  v_f  u,v_c  v_f
  |           |           |
  |           |           |
  +----u_f----+----u_f----+


  If psi is defined at cell nodes then
  cell faces correspond to a standard C-grid


  Parameters
  ----------

  psi : array [(nz,) ny,nx]
  Delta: float
  bc: pad psi field with boundary conditions
  loc: 'center' or 'faces' (default center)

  Returns
  -------
  
  size of returned arry depend on loc:
  if center -> ny,nx
  if faces -> ny+1,nx and ny,nx+1

  u: array [nz, ny(+1),nx]
  v: array [nz, ny,nx(+1)]
  '''

  psi_pad = pad_bc(psi, bc=bc)

  if loc == 'center' or loc == 'node':
    u = (psi_pad[...,:-2,1:-1] - psi_pad[...,2:,1:-1])/(2*Delta)
    v = (psi_pad[...,1:-1,2:] - psi_pad[...,1:-1,:-2])/(2*Delta)
  elif loc == 'faces':
    u = (psi_pad[...,:-1,1:-1] - psi_pad[...,1:,1:-1])/Delta
    v = (psi_pad[...,1:-1,1:] - psi_pad[...,1:-1,:-1])/Delta
    
  return u,v


def comp_ke(psi, Delta):

  '''
  Compute KE at cell center

  KE =  (u^2 + v^2)/2

  Parameters
  ----------

  psi : array [(nz,) ny,nx]
  Delta: float

  Returns
  -------

  KE: array [(nz,) ny,nx]

  '''

  f_type = field_type(psi)

  # need to interpolate u^2 and v^2 at cell center and *not* u and v
  # So that if we compare the integral of ke and the integral of 0.5*q*p
  # we get the same answer within machine precision

  if f_type == 'center':
    u,v = comp_vel(psi, Delta, bc='dirichlet', loc='faces')

    ke = 0.25*(v[...,:,1:]**2 + v[...,:,:-1]**2 +
               u[...,1:,:]**2 + u[...,:-1,:]**2)
  else:
    # pad psi with 0 ('dirichlet_face') to get (zero) cross-boundary values for
    # u and v. We remove this padding in the computation of ke (1:-1 indices)
    u,v = comp_vel(psi, Delta, bc='dirichlet_face', loc='faces')

    ke = 0.25*(u[...,1:-1,1:]**2 + u[...,1:-1,:-1]**2 +
               v[...,1:,1:-1]**2 + v[...,:-1,1:-1]**2)


  return ke


def comp_pe(psi, dh,N2,f0,Delta):

  '''
  Compute KE at cell center

  PE =  b^2/(2 N^2)

  Parameters
  ----------

  psi : array [nz, ny,nx]
  dh : array [nz]
  N2 : array [nz, (ny,nx)]
  f0 : scalar or array [ny,nx]
  Delta: float

  Returns
  -------

  PE: array [nz-1, ny,nx]

  '''

  nd = psi.ndim
  si = psi.shape
  N = si[-1]
  if nd != 3:
    print("dimension of psi should be [nz, ny,nx]")
    sys.exit(1)

  N2,f0 = reshape3d(dh,N2,f0)

  b = p2b(psi,dh,f0)
  pe = 0.5*b**2/N2

  return pe


def integral(psi, dh, Delta, average=False):

  '''
  Compute integral of a field

  if the field is defined on p-levels, the integral is the usual integral
  if the field is defined on b-levels, we assume that psi = 0 at
  the upper and lower boundaries


  **if psi is a node field, we assume psi=0 at the boundary**

  Parameters
  ----------

  psi : array [nz, ny,nx] for p-level field or array [nz-1, ny,nx] for b-level field
  dh : array [nz], *dh is the layer thickness of the p-level* (no matter what)
  Delta: float
  average: if True: divide the integral by the total volume (default is False)

  Returns
  -------

  psi_i = scalar

  '''
  si = psi.shape
  nl0 = si[0]
  N = si[-1]
  nl = len(dh)

  # total H even for b-levels
  Ht = np.sum(dh)

  dhl = np.copy(dh)
  if nl0 == nl-1:
    dhl = 0.5*(dh[1:] + dh[:-1])

  if average:
    psi_i = np.sum(psi*dhl[:,None,None])/Ht/N**2
  else:
    psi_i = np.sum(psi*dhl[:,None,None])*Delta*Delta

  return psi_i

def intz(psi, dh):
    '''
    Compute vertical integral of a field

    if the field is defined on p-levels, the integral is the usual integral
    if the field is defined on b-levels, we assume that psi = 0 at
    the upper and lower boundaries


    Parameters
    ----------

    psi : array [nz, ny,nx] for p-level field or array [nz-1, ny,nx] for b-level field
    dh : array [nz], *dh is the layer thickness of the p-level* (no matter what)

    Returns
    -------

    psi_out: array [ny,nx]

    '''
    si = psi.shape
    nl0 = si[0]
    N = si[-1]
    nl = len(dh)

    # total H even for b-levels
    Ht = np.sum(dh)

    dhl = np.copy(dh)
    if nl0 == nl-1:
        dhl = 0.5*(dh[1:] + dh[:-1])

    if psi.ndim == 3:
        psi_out = np.sum(psi*dhl[:,None,None], axis=0)
    elif psi.ndim == 2:
        psi_out = np.sum(psi*dhl[:,None], axis=0)

    return psi_out



def lorenz_cycle(pfiles,dh,N2,f0,Delta,bf=0, nu=0, nu4=0, forcing_z=0, forcing_b=0, toc=0, nu_in_b=True, bc_fac=0, interp=False, average=False, maps=False, spec_flx=False):
  '''
  Compute Lorenz energy cycle

  Parameters
  ----------

  pfiles : list of pressure files
  dh : array [nz] 
  N2 : array [nz (,ny,nx)]
  f0 : scalar or array [ny,nx]
  Delta: float
  bf : scalar  (bottom friction coef = d_e *f0/(2*dh[-1]) with d_e the thickness 
  of the bottom Ekman layer, or bf = Ekb/(Rom*2*dh[-1]) with non dimensional params)
  nu: scalar, harmonic viscosity
  nu4: scalar, bi-harmonic viscosity
  forcing_z : array [ny,nx] wind forcing (exactly the same as the rhs of the PV eq.)
              or list of files
  forcing_b : array [ny,nx]  =(buoyancy forcing)/N2 (entoc), or list of files
              if list of files and toc !=0, load sst to recompute entoc offline
              if list of files and toc=0, then forcing_b reads entoc
  toc :  array[nz] temperature anomaly (only used for buoyancy forcing)
  nu_in_b : Bool,
            if QG equations are derived from PE there is dissip in PE (nu_in_b=True)        
            if QG equations are derived from SW there is no thichness dissipation (nu_in_b=False)
  bc_fac: scalar (only used if psi is a node field) 0 for free slip or 1 for no slip 
  average : Bool, if False, return energy integral (default), if True: return energy average
  maps: Bool, if True (default=False), return depth integrated maps for each terms, 
            including both 'perspectives (i.e. EKE and MKE)' of eddy-mean flow energy transfers.
  spec_flx: Bool, if True (default=False), return (some) spetral fluxes

  Returns
  -------

  lec: dict of all energy fluxes and energy reservoirs. Sign convention matches name:
    e.g. if mke2mpe >0 then there is a transfer from mke to mpe
  '''

  N2,f0 = reshape3d(dh,N2,f0)

  si_t = read_time(pfiles)

  p = load_generic(pfiles, 0, 'p', 1/f0, interp=interp, si_t=si_t)
  nl,N,naux = p.shape
  
  if not isinstance(forcing_z, list):
    loc_forcing_z = np.copy(forcing_z)

  if not isinstance(forcing_b, list):
    loc_forcing_b = np.copy(forcing_b)
  
  # compute mean
  p_me = np.zeros((nl,N,N))
  w_me = np.zeros((nl-1,N,N))
  f_me = np.zeros((N,N))
  d_me = np.zeros((N,N))

  n_me = 1
  for it in range(0,si_t):

    #print("Loop 1/2, iter ", it, "/", si_t-1, end="\r")
    print("Loop 1/2, iter ", it, "/", si_t-1)
  
    p = load_generic(pfiles, it, 'p', rescale=1/f0, interp=interp, si_t=si_t, subtract_bc=True)
    if isinstance(forcing_z, list):
      loc_forcing_z = load_generic(forcing_z, it, 'wekt', rescale=f0/dh[0], interp=(not interp), si_t=si_t)
      if (not interp):
        loc_forcing_z = pad_bc(loc_forcing_z, bc='neumann')
    if isinstance(forcing_b, list):
      if isinstance(toc,int):
        loc_forcing_b = load_generic(pfiles, it, 'entoc', interp=interp, si_t=si_t)
      else:
        sst = load_generic(forcing_b, it, 'sst', interp=(not interp), si_t=si_t)
        if (not interp):
          sst = pad_bc(sst, bc='neumann')
        wekt = loc_forcing_z*dh[0]/f0 # remove scaling
        entoc = -0.5*wekt*( sst - toc[0] ) /(toc[0]-toc[1])
        loc_forcing_b = entoc - np.mean(entoc)


    w = get_w(p,dh, N2[:,0,0],f0[0,0], Delta, bf,loc_forcing_z, loc_forcing_b, nu=(not nu_in_b)*nu, nu4=(not nu_in_b)*nu4, bc_fac=bc_fac)
  
    p_me += (p - p_me)/n_me
    w_me += (w - w_me)/n_me
    f_me += (loc_forcing_z - f_me)/n_me
    d_me += (loc_forcing_b - d_me)/n_me
    n_me += 1
  
  z_me = laplacian(p_me,Delta, bc_fac=bc_fac)
  b_me = p2b(p_me, dh, f0)
  s_me = p2stretch(p_me,dh, N2,f0)
  q_me = p2q(p_me, dh, N2,f0, Delta)
  ke_me = comp_ke(p_me,Delta)
  pe_me = comp_pe(p_me, dh, N2,f0, Delta)
  
  ei_ke_me = intz(ke_me, dh)
  ei_pe_me = intz(pe_me, dh)
  
  e_surf   = np.zeros((nl,N,N))
  e_bottom = np.zeros((nl,N,N))
  e_diab   = np.zeros((nl-1,N,N))
  
  dissip_k_me = -nu4*laplacian(laplacian(z_me,Delta, bc_fac=bc_fac),Delta, bc_fac=bc_fac)
  dissip_p_me = -nu4*laplacian(laplacian(s_me,Delta),Delta)
  dissip_k_me += nu*laplacian(z_me,Delta, bc_fac=bc_fac)
  dissip_p_me += nu*laplacian(s_me,Delta)
  
  
  bottom_ekman = -bf*laplacian(p_me[-1,:,:],Delta, bc_fac=bc_fac)
  
  e_surf[0,:,:] = -p_me[0,:,:]*f_me
  e_bottom[-1,:,:] = -p_me[-1,:,:]*bottom_ekman
  
  e_diab[0,:,:] = b_me[0,:,:]*d_me

  ei_surf_me   = intz(e_surf, dh)
  ei_bottom_me = intz(e_bottom, dh)
  ei_diss_k_me = intz(-p_me*dissip_k_me, dh)
  ei_diss_p_me = intz(-p_me*dissip_p_me, dh)
  ei_wb_me     = intz(w_me*b_me, dh)
  ei_diab_me   = intz(e_diab, dh)
  
  # compute all terms
  ei_ke           = np.zeros((N-1, N-1))
  ei_pe           = np.zeros((N, N))
  ei_surf         = np.zeros((N, N))
  ei_bottom       = np.zeros((N, N))
  ei_diss_k       = np.zeros((N, N))
  ei_diss_p       = np.zeros((N, N))
  ei_wb           = np.zeros((N, N))
  ei_ke_me2ke_p   = np.zeros((N, N))
  ei_pe_me2pe_p   = np.zeros((N, N))
  ei_ke_me2ke_p_2 = np.zeros((N, N))
  ei_pe_me2pe_p_2 = np.zeros((N, N))
  ei_diab         = np.zeros((N, N))
  # spectral fluxes
  if spec_flx:
    epe2eke_spflx   = np.zeros([int((N-1)/2)])
    jpz_m = np.zeros([nl, N, N])
    jps_m = np.zeros([nl, N, N])
  
  n_me = 1
  for it in range(0,si_t):
    #print("Loop 2/2, iter ", it, "/", si_t-1, end="\r")
    print("Loop 2/2, iter ", it, "/", si_t-1)

    p = load_generic(pfiles, it, 'p', 1/f0, interp=interp, si_t=si_t, subtract_bc=interp)
    if isinstance(forcing_z, list):
      loc_forcing_z = load_generic(forcing_z, it, 'wekt', f0/dh[0], interp=(not interp), si_t=si_t)
      if (not interp):
        loc_forcing_z = pad_bc(loc_forcing_z, bc='neumann')
    if isinstance(forcing_b, list):
      if isinstance(toc,int):
        loc_forcing_b = load_generic(pfiles, it, 'entoc', interp=interp, si_t=si_t)
      else:
        sst = load_generic(forcing_b, it, 'sst', interp=(not interp), si_t=si_t)
        if (not interp):
          sst = pad_bc(sst, bc='neumann')
        wekt = loc_forcing_z*dh[0]/f0 # remove scaling
        entoc = -0.5*wekt*( sst - toc[0] ) /(toc[0]-toc[1])
        loc_forcing_b = entoc - np.mean(entoc)

    z = laplacian(p,Delta, bc_fac=bc_fac)
    b = p2b(p, dh, f0)
    s = p2stretch(p,dh, N2,f0)
    w = get_w(p,dh, N2[:,0,0],f0[0,0], Delta, bf,loc_forcing_z, forcing_b, nu=(not nu_in_b)*nu, nu4=(not nu_in_b)*nu4, bc_fac=bc_fac)
    q = p2q(p, dh, N2,f0, Delta)
    ke = comp_ke(p,Delta)
    pe = comp_pe(p, dh, N2,f0, Delta)
  
    p_p = p - p_me
    z_p = z - z_me
    b_p = b - b_me 
    s_p = s - s_me
    w_p = w - w_me
    q_p = q - q_me
    ke_p = ke - ke_me
    pe_p = pe - pe_me
    
    jpz = jacobian(p_p,z_p, Delta)
    jps = jacobian(p_p,s_p, Delta)
    jpz_2 = jacobian(p_p,z_me, Delta)
    jps_2 = jacobian(p_p,s_me, Delta)
  
    ke_me2ke_p = -p_me*jpz
    pe_me2pe_p =  p_me*jps	#remove '-' sign ; =-b.J(\psi, b) through integration by part
    ke_me2ke_p_2 = -p_p*jpz_2
    pe_me2pe_p_2 =  p_p*jps_2	#remove '-' sign ; =-b.J(\psi, b) through integration by part
    
    dissip_k = -nu4*laplacian(laplacian(z_p,Delta, bc_fac=bc_fac),Delta, bc_fac=bc_fac)
    dissip_p = -nu4*laplacian(laplacian(s_p,Delta),Delta)
    dissip_k += nu*laplacian(z_p,Delta, bc_fac=bc_fac)
    dissip_p += nu*laplacian(s_p,Delta)

    bottom_ekman = -bf*laplacian(p_p[-1,:,:],Delta, bc_fac=bc_fac)
  
    e_surf[0,:,:] = -p_p[0,:,:]*(loc_forcing_z - f_me)
    e_bottom[-1,:,:] = -p_p[-1,:,:]*bottom_ekman

    e_diab[0,:,:] = b_p[0,:,:]*(loc_forcing_b - d_me)
    
    ei_ke_me2ke_p   += (intz(ke_me2ke_p, dh) - ei_ke_me2ke_p) / n_me
    ei_pe_me2pe_p   += (intz(pe_me2pe_p, dh) - ei_pe_me2pe_p) / n_me
    ei_ke_me2ke_p_2 += (intz(ke_me2ke_p_2, dh) - ei_ke_me2ke_p_2) / n_me
    ei_pe_me2pe_p_2 += (intz(pe_me2pe_p_2, dh) - ei_pe_me2pe_p_2) / n_me
    ei_surf         += (intz(e_surf, dh) - ei_surf) / n_me
    ei_bottom       += (intz(e_bottom, dh) - ei_bottom) / n_me
    ei_diss_k       += (intz(-p_p*dissip_k, dh) - ei_diss_k) / n_me
    ei_diss_p       += (intz(-p_p*dissip_p, dh) - ei_diss_p) / n_me
    ei_wb           += (intz(w_p*b_p, dh) - ei_wb) / n_me
    ei_ke           += (intz(ke_p, dh) - ei_ke) / n_me
    ei_pe           += (intz(pe_p, dh) - ei_pe) / n_me
    ei_diab         += (intz(e_diab, dh) - ei_diab) / n_me
    
    #-- compute some spectral fluxes --
    if spec_flx:
        kkk, tmp_flx  = get_spec_flux(w_p, b_p, Delta, window=None)
        epe2eke_spflx += ( intz(tmp_flx, dh) - epe2eke_spflx ) / n_me
        jpz_m += (jpz - jpz_m)/n_me
        jps_m += (jps - jps_m)/n_me
    #
    n_me += 1

  # sign convention matches name
  lec = {}
  lec["f2mke"]   = ei_surf_me.sum()*Delta**2             
  lec["f2eke"]   = ei_surf.sum()*Delta**2
  lec["f2mpe"]   = ei_diab_me .sum()*Delta**2            
  lec["f2epe"]   = ei_diab.sum()*Delta**2
  lec["mke2mpe"] = -ei_wb_me.sum()*Delta**2              
  lec["epe2eke"] = ei_wb.sum()*Delta**2         
  lec["mke2eke"] = ei_ke_me2ke_p.sum()*Delta**2
  lec["mpe2epe"] = ei_pe_me2pe_p.sum()*Delta**2 
  lec["eke2mke"] = ei_ke_me2ke_p_2.sum()*Delta**2
  lec["epe2mpe"] = ei_pe_me2pe_p_2.sum()*Delta**2
  lec["mke2dis"] = -ei_diss_k_me.sum()*Delta**2          
  lec["eke2dis"] = -ei_diss_k.sum()*Delta**2    
  lec["mpe2dis"] = -ei_diss_p_me.sum()*Delta**2          # overwritten if not nu_in_b
  lec["epe2dis"] = -ei_diss_p.sum()*Delta**2    # overwritten if not nu_in_b
  lec["mke2bf"]  = -ei_bottom_me.sum()*Delta**2          
  lec["eke2bf"]  = -ei_bottom.sum()*Delta**2    
  lec["mke"]     = ei_ke_me.sum()*Delta**2      
  lec["eke"]     = ei_ke.sum()*Delta**2
  lec["mpe"]     = ei_pe_me.sum()*Delta**2      
  lec["epe"]     = ei_pe.sum()*Delta**2
  # maps
  if maps:
    lec["f2mke_map"]   = ei_surf_me             
    lec["f2eke_map"]   = ei_surf
    lec["f2mpe_map"]   = ei_diab_me             
    lec["f2epe_map"]   = ei_diab
    lec["mke2mpe_map"] = -ei_wb_me              
    lec["epe2eke_map"] = ei_wb         
    lec["mke2eke_map"] = ei_ke_me2ke_p
    lec["mpe2epe_map"] = ei_pe_me2pe_p 
    lec["eke2mke_map"] = ei_ke_me2ke_p_2
    lec["epe2mpe_map"] = ei_pe_me2pe_p_2
    lec["mke2dis_map"] = -ei_diss_k_me          
    lec["eke2dis_map"] = -ei_diss_k    
    lec["mpe2dis_map"] = -ei_diss_p_me          # overwritten if not nu_in_b
    lec["epe2dis_map"] = -ei_diss_p    # overwritten if not nu_in_b
    lec["mke2bf_map"]  = -ei_bottom_me          
    lec["eke2bf_map"]  = -ei_bottom    
    lec["mke_map"]     = ei_ke_me      
    lec["eke_map"]     = ei_ke
    lec["mpe_map"]     = ei_pe_me      
    lec["epe_map"]     = ei_pe
  # spectral fluxes
  if spec_flx:
    lec["k"]        = kkk
    lec["epe2eke_spflx"]  = epe2eke_spflx
    kkk, tmp_flx  = get_spec_flux(-p_me, jpz_m, Delta, window=None)
    lec["mke2eke_spflx"]  = intz(tmp_flx, dh)
    kkk, tmp_flx = get_spec_flux(-p_me, jps_m, Delta, window=None)
    lec["mpe2epe_spflx"]  = intz(tmp_flx, dh)

  if not nu_in_b:
    lec["mpe2dis"] = 0
    lec["epe2dis"] = 0
  if not isinstance(forcing_z, list):
    lec["f2eke"]   = 0
  if not isinstance(forcing_b, list):
    lec["f2mpe"]   = 0 # not strictly speaking in this if
    lec["f2epe"]   = 0

  lec["average"] = average

  print("Done\n")

  return lec


def draw_lorenz_cycle(lec, us=1, ts=1, rho=1):

  '''
  Draw Lorenz energy cycle

  To help visualisation, one can use us, ts and rho: 
  - energies are multiplied by rho/us^2
  - and fluxes are multiplied by rho*ts/us^2 (amount of energy added/removed in ts)

  if lec["average"] == True, energies from lorenz_cycle are m^5/s^2, fluxes are m^5/s^3
  
  When multiplied by rho, energies are in joules and fluxes in watt. You can use
  us and ts to get multiples of joule and watts 
  for instance to get energies in EJ (exajoules) and fluxes in GW (gigawatt), set
  - us = 1e9 (divide energy by 1e18) 
  - ts = 1e9 (divide flux by 1e9) 
  

  if lec["average"] == True, energies are m^2/s^2, fluxes are m^2/s^3

  us and ts are just velocity scales and time scales.
  *no need to set rho in that case*


  Parameters
  ----------

  lec: dict of all energy fluxes and energy reservoirs from lorenz_cycle function
  us: float, velocity scale factor
  ts: float, time scale factor
  rho: float, density

  Returns
  -------

  nothing

  '''

  f2mke   = lec["f2mke"]*rho*ts/us**2  
  f2eke   = lec["f2eke"]*rho*ts/us**2  
  f2mpe   = lec["f2mpe"]*rho*ts/us**2
  f2epe   = lec["f2epe"]*rho*ts/us**2  
  mke2mpe = lec["mke2mpe"]*rho*ts/us**2
  epe2eke = lec["epe2eke"]*rho*ts/us**2
  mke2eke = lec["mke2eke"]*rho*ts/us**2
  mpe2epe = lec["mpe2epe"]*rho*ts/us**2
  mke2dis = lec["mke2dis"]*rho*ts/us**2
  eke2dis = lec["eke2dis"]*rho*ts/us**2
  mpe2dis = lec["mpe2dis"]*rho*ts/us**2
  epe2dis = lec["epe2dis"]*rho*ts/us**2
  mke2bf  = lec["mke2bf"]*rho*ts/us**2 
  eke2bf  = lec["eke2bf"]*rho*ts/us**2
  mke     = lec["mke"]*rho/us**2    
  eke     = lec["eke"]*rho/us**2    
  mpe     = lec["mpe"]*rho/us**2    
  epe     = lec["epe"]*rho/us**2

  plt.figure()
  plt.text(0.5,0.5  ,"MKE\n {0:0.0f}".format(mke),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(-0.5,0.5 ,"MPE\n {0:0.0f}".format(mpe),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(-0.5,-0.5,"EPE\n {0:0.0f}".format(epe),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(0.5,-0.5 ,"EKE\n {0:0.0f}".format(eke),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  
  # wb
  wb_sign = np.sign(mke2mpe)
  plt.arrow(wb_sign*0.25,0.5,-wb_sign*0.5,0,width=0.01, length_includes_head=True)
  plt.text(0,0.6, "{0:0.0f}".format(np.abs(mke2mpe)),horizontalalignment='center', verticalalignment='center')
  
  wb_sign = np.sign(epe2eke)
  plt.arrow(-wb_sign*0.25,-0.5,wb_sign*0.5,0,width=0.01, length_includes_head=True)
  plt.text(0,-0.4, "{0:0.0f}".format(np.abs(epe2eke)),horizontalalignment='center', verticalalignment='center')
  
  # mean to eddy
  k2k_sign = np.sign(mke2eke)
  plt.arrow(0.5,k2k_sign*0.25,0,-k2k_sign*0.5,width=0.01, length_includes_head=True)
  plt.text(0.6,0, "{0:0.0f}".format(np.abs(mke2eke)))
  
  p2p_sign = np.sign(mpe2epe)
  plt.arrow(-0.5,p2p_sign*0.25,0,-p2p_sign*0.5,width=0.01, length_includes_head=True)
  plt.text(-0.8,0, "{0:0.0f}".format(np.abs(mpe2epe)))
  
  
  # forcing
  plt.arrow(0.5,1.25,0,-0.5,width=0.01, length_includes_head=True)
  plt.text(0.55,1, "W:{0:0.0f}".format(f2mke))
  
  if f2eke != 0:  
    f2e_sign = np.sign(f2eke)
    plt.arrow(0.5,-1.25 + 0.25*(1-f2e_sign),0,f2e_sign*0.5,width=0.01, length_includes_head=True)
    plt.text(0.55,-1, "W:{0:0.0f}".format(np.abs(f2eke)))

  if f2epe != 0:  
    f2e_sign = np.sign(f2epe)
    plt.arrow(-0.5,-1.25 + 0.25*(1-f2e_sign),0,f2e_sign*0.5,width=0.01, length_includes_head=True)
    plt.text(-0.8,-1, "E:{0:0.0f}".format(np.abs(f2epe)))

  if f2mpe != 0:  
    f2e_sign = np.sign(f2mpe)
    plt.arrow(-0.5,1.25 - 0.25*(1-f2e_sign),0,-f2e_sign*0.5,width=0.01, length_includes_head=True)
    plt.text(-0.8,1, "E:{0:0.0f}".format(np.abs(f2mpe)))

  # viscous dissip
  plt.arrow(0.75,0.6,0.5,0,width=0.01, length_includes_head=True)
  plt.text(1,0.65, "D:{0:0.0f}".format(mke2dis),horizontalalignment='center')
  
  plt.arrow(0.75,-0.4,0.5,0,width=0.01, length_includes_head=True)
  plt.text(1,-0.35, "D:{0:0.0f}".format(eke2dis),horizontalalignment='center')
  
  if mpe2dis > 0:
    plt.arrow(-0.75,0.5,-0.5,0,width=0.01, length_includes_head=True)
    plt.text(-1,0.55, "D:{0:0.0f}".format(mpe2dis),horizontalalignment='center')
  
  if epe2dis > 0:
    plt.arrow(-0.75,-0.5,-0.5,0,width=0.01, length_includes_head=True)
    plt.text(-1,-0.45, "D:{0:0.0f}".format(epe2dis),horizontalalignment='center')
  
  # Bottom friction
  plt.arrow(0.75,0.4,0.5,0,width=0.01, length_includes_head=True)
  plt.text(1,0.25, "BF:{0:0.0f}".format(mke2bf),horizontalalignment='center')
  
  plt.arrow(0.75,-0.6,0.5,0,width=0.01, length_includes_head=True)
  plt.text(1,-0.75, "BF:{0:0.0f}".format(eke2bf),horizontalalignment='center')
  
  
  plt.xlim([-1.5,1.5])
  plt.ylim([-1.5,1.5])
       
  plt.show()

  return
