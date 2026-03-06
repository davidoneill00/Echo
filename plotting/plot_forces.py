
import numpy as np
from read_h5 import load
import matplotlib.pyplot as plt
import math
import argparse


    
def Ostriker99(v,c,t,rmin,tmin, GM=1, rho0 = 1):
    """
    Return the drag force as predicted by Ostriker (1999) for rectilinear motion for a given Mach number, velocity and time.
    """
    F  = 4 * np.pi * GM**2 * rho0 / v**2
    x1 = 0.97
    x2 = 1.02
    M  = v/c
    if M<=x1:
        return [-F * (0.5*math.log((1+M)/(1-M))-M),0]
    elif M>=x2:
        Lg   = math.log(v*(t-tmin)/rmin)
        return [-F * (0.5*math.log(1-1/(M**2))+Lg),0]
    else:
        c1 = v/x1
        c2 = v/x2
        y1 = Ostriker99(v,c1,t,rmin,tmin)[0]
        y2 = Ostriker99(v,c2,t,rmin,tmin)[0]
        return [y1+(M-x1)*(y2-y1)/(x2-x1),0]






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str  , help="Path to the checkpoint file")
    parser.add_argument("--vmin"    , type=float, help="Minimum value for the field", default=-3)
    parser.add_argument("--vmax"    , type=float, help="Maximum value for the field", default= 0)
    parser.add_argument("--cmap"    , type=str  , help="Colormap for the plot"      , default="jet")
    args = parser.parse_args()

    Domain, Trajectory, Params, Alpha, Alpha_t, Levels = load(args.checkpoint)


    print(Trajectory.Force)
    plt.figure()
    plt.plot(Trajectory.Force_t   , -Trajectory.Force[:,0])
    plt.plot(Trajectory.Force_t   , -Trajectory.Force[:,1])
    plt.plot(Trajectory.Force_t   , -Trajectory.Force[:,2])
    plt.scatter(Trajectory.Force_t, -Trajectory.Force[:,0])
    plt.scatter(Trajectory.Force_t, -Trajectory.Force[:,1])
    plt.scatter(Trajectory.Force_t, -Trajectory.Force[:,2])
    plt.plot(Trajectory.Force_t, [Ostriker99(v=2.0, c=1.0, t=t, rmin=0.1, tmin=0.0, GM=1, rho0 = 1)[0] for t in Trajectory.Force_t], c = 'black', linestyle = 'dashed')
    #plt.plot(Trajectory.Force_t, [2 * Trajectory.Force[i,0] / Ostriker99(v=2.0, c=1.0, t=Trajectory.Force_t[i], rmin=0.1, tmin=0.0, GM=1, rho0 = 1) for i in range(0,len(Trajectory.Force_t))])
    plt.savefig("/Users/davidoneill/Desktop/Echo/Outputs/Forces.png", dpi = 400)
