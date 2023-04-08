from collections import namedtuple
from os import times

from scipy.interpolate import interp1d
import numpy as np

import numba

Calibration = namedtuple(
    'calibration', 
    'width depth mean_water_level_difference manning_coefficient'
    )

# def Qmec(calibration, timestamps, water_levels, distances, dt):
#     assert len(water_levels) == 2, 'Current implementation supports only 2 water levels: upstream and downstream'
#     assert len(distances) == 1, 'A single distance must be provided with 2 stations'

#     return Qmec_2_stations(calibration, timestamps, water_levels[0], water_levels[1], distances[0], dt)

def Qmec(calibration, mtime_h, h1, h2, dx, dt):
    '''
    Solves the following differential river equation:

    dQ/dt = - g B_e (h_e + h) (dh + dzeta)/dx 
            - g n^2{[B_e + 2(h_e + h)]/[B_e(h_e + h)]}^(4/3) Q|Q|/[B_e(h_e + h)] 
            + 2Q/(h_e + h)(dh/dt)

    INPUT:
            
        calibration: a namedtuple with the following fields:
                    - width: the effective width B_e [m]
                    - depth: the effective depth h_e [m]
                    - mean_water_level_difference: the hydraulic slope dzeta [m]
                    - manning_coefficient: the effective Manning's coefficient [s m^(-1/3)]

        mtime_h: The time of the water level time series h1 and h2 expressed 
                 in unix timestamp should be a nparray type

        h1:      The water level time series at the upstream limit [m] should be a nparray type
        h2:      The water level time series at the downstream limit [m] should be a nparray type
        dx:      The distance between the upstream and downstream limits [m]
        dt:      The time step [s]

    OUTPUT:

        mtime_Q: The time of the computed discharge time series expressed 
                 in days relative to 1 Jan 0000 00:00:00 

        Q:       The computed discharge time-series [m^3/s]

    LAST REVISION: 18 December 2019

    REFERENCE:
        Bourgault D and Matte P (2019), A physically-based method for 
            real-time monitoring of tidal river discharges from water level 
            observations with an application to the St.~Lawrence River, 
            submitted to the Journal of Geophysical Research-Oceans

    CONTACTS: 
        - Daniel Bourgault (daniel_bourgault@uqar.ca) 
        - Pascal Matte (pascal.matte@canada.ca) 
     '''

    ## Spin-up time
    # This is the time it takes the model to reach periodic steady state. 
    # This time is used to remove (i.e. to replace with NaNs) the number of 
    # data points after model initialization. This happens not only at initial 
    # time but also every time missing data are encountered in the input
    # water levels.
    spinup_time = 7*3600                # The model spin-up time [s]
    N_spinup    = round(spinup_time/dt) # The number of time steps given dt

    # Start and end of the initial spinup.
    spinup_start = 0
    spinup_end = N_spinup - 1

    # Change of variable names for brevity
    Be    = calibration.width
    he    = calibration.depth
    dzeta = calibration.mean_water_level_difference
    ne    = calibration.manning_coefficient
    
    ## Construct the time vector
    # dmtime   = dt/86400  # Convert the dt [s] into fraction of a day dmtime
    # mmax = int((mtime_h[-1] - mtime_h[0]) / dmtime) + 1  # Total number of time steps
    # mtime_Q = np.linspace(mtime_h[0], mtime_h[-1], mmax)

    dmtime  = dt/86400
    mtime_Q = np.arange(0, mtime_h[-1] - mtime_h[0] + dmtime, dmtime)
    mtime_Q = mtime_h[0] + mtime_Q 
    mmax    = len(mtime_Q)

    # Interpolate the water level at the model time
    interp_method = 'linear'    # Important to leave it linear. Long-term results are sensitive to this.

    # ORIGINAL CODE 
    # h1fct = interp1d(mtime_h, h1, kind=interp_method)
    # h1i = h1fct(mtime_Q)
    # h1fct = interp1d(mtime_h, h2, kind=interp_method)
    # h2i = h2fct(mtime_Q)
    
    # NEW CODE 
    h1fct = interp1d((mtime_h - np.mean(mtime_h)) / np.std(mtime_h), h1, kind=interp_method)
    h1i = h1fct((mtime_Q - np.mean(mtime_h)) / np.std(mtime_h))
    h2fct = interp1d((mtime_h - np.mean(mtime_h)) / np.std(mtime_h), h2, kind=interp_method)
    h2i = h2fct((mtime_Q - np.mean(mtime_h)) / np.std(mtime_h))

    # Intermediate calculations
    dh = h2i - h1i            # Difference in water level between the two stations
    hm = (h1i + h2i)/2        # Mean water level between the two stations

    # Geometry parameters for a rectangular cross-section
    # expanding the Be value to a vector of mmax length
    Be = Be * np.ones(mmax)   # Constant width
    hmTmp = hm + he
    Ae = Be * hmTmp           # Mean cross sectional area
    Pe = Be + 2*(hmTmp)       # Wetted perimeter
    Re = Ae/Pe                # Hydraulic radius
 
    # Initial condition
    Q = np.zeros(mmax)

    # Main calculation and time loop
    Q = iter_qmec(Q, hm, Ae, Re, Be, dh, dzeta, ne, mmax, N_spinup, spinup_start, spinup_end, dx, dt)

    return mtime_Q, Q, h1i, h2i

@numba.jit(nopython=True)
def iter_qmec(Q, hm, Ae, Re, Be, dh, dzeta, ne, mmax, N_spinup, spinup_start, spinup_end, dx, dt):
    
    g = 9.81 # Gravitationl acceleration (m/s^2)

    for m in range(1, mmax): # mmax
        # First-order scheme for the first time-step with Q(1) = 0 or for the
        # first time-step following missing water level data with Q(m-1) = 0
        
        if (m == 1 or not np.isfinite(Q[m-1])) or (m > 1 and not np.isfinite(Q[m-2])):
            pressure_gradient = -g*(Ae[m-1]*(dh[m-1] + dzeta)/dx)
            Q[m] = dt*pressure_gradient
        else:            
            dhdt1 = (hm[m] - hm[m-2])/(2*dt)
            if m == 2:
                dhdt2 = (hm[m-1] - hm[m-2])/dt
            else:
                dhdt2 = (hm[m-1] - hm[m-3])/(2*dt)

            # Second-order Adams-Bashforth scheme
            pressure_gradient = (3/2)*(-g*Ae[m-1]*(dh[m-1] + dzeta)/dx) - (1/2)*(-g*Ae[m-2]*(dh[m-2] + dzeta)/dx)

            bottom_friction   = (3/2)*(-g*((ne**2)/(Re[m-1]**(4/3)))*Q[m-1]*abs(Q[m-1])/Ae[m-1]) - (1/2)*(-g*((ne**2)/(Re[m-2]**(4/3)))*Q[m-2]*abs(Q[m-2])/Ae[m-2])

            advection         = (3/2)*(2*(Q[m-1]/Ae[m-1])*Be[m-1]*dhdt1) -(1/2)*(2*(Q[m-2]/Ae[m-2])*Be[m-2]*dhdt2)

            Q[m] = Q[m-1] + dt*(pressure_gradient + bottom_friction + advection)

        # Set the start and end spinup indices if the current Q isn't finite.
        if not np.isfinite(Q[m]):
            if m > spinup_end:
                spinup_start = m
            spinup_end = min(m + N_spinup - 1, mmax - 1)

        # Replace the discharge calculated during the spin-up phase
        # (of duration spinup_time) with NaN
        if m == spinup_end + 2 or (m == mmax - 1 and (m == spinup_end or m == spinup_end + 1)):
            Q[spinup_start:spinup_end + 1] = np.nan

    return Q

    if __name__ == "__main__":
        pass
