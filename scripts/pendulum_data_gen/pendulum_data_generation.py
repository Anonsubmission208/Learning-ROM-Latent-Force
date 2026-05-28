import argparse
import numpy as np
import math
from pathlib import Path
import pickle

GRAVITY = 9.8

def draw_line(x0, y0, x1, y1, canvas):    
    # Round input coordinates to the nearest integers for starting points
    x0, y0 = round(x0), round(y0)
    x1, y1 = round(x1), round(y1)
    
    # Calculate the difference between the points
    dx = abs(x1 - x0) 
    dy = abs(y1 - y0)
    
    # Determine the direction of the line
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    # Error term
    err = dx - dy
    
    while True:
        # Add the current integer pixel coordinate to the list
        canvas[y0, x0] = 1        
        # Check if the endpoint is reached
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def rasterize_frames(x, y, px, py, radius):
        # generate array ranging from 0 to resolutions        
        sq_x = (np.arange(px) - x)**2
        sq_y = (np.arange(py) - y)**2

        # add dim to x and y arrays and add to get (batch, vert_res, vert_res)
        sq = sq_x[:,None,:] + sq_y[:,:,None]
    
        # turn pixels on or off depending on if they are less than the radius squared
        images = 1 * (sq < radius*radius)

        x = list(np.squeeze(x))
        y = list(np.squeeze(y))
                 
        for i in range(len(images)):
            draw_line(px/2, 0, x[i], y[i], images[i]) 
      
        return images    

def generate_video_frames(x_pos, y_pos, px, py, radius):   
    video = [rasterize_frames(x, y, px, py, radius) for x, y in zip(x_pos, y_pos)]
    return np.array(video)

def theta_to_pos(thetas, px, py, l, pad):
    x = l * np.sin(thetas)*min(px, py)*pad + px/2
    y = l * np.cos(thetas)*min(px, py)*pad
    return x, y

def kernel_sq(X,Y, l=1, var=1):
    return var*np.exp(-0.5*(X[:,None]-Y[None,:])**2/(l**2))

def sample_wind(n, time, dt, length_scale, variance):
    step_size = int(time/dt)
    
    gp_time = np.linspace(0, time, step_size)
    wind = np.zeros((n, step_size, 1))

    cov = kernel_sq(gp_time, gp_time, l=length_scale, var=variance)
    mean = np.zeros(len(gp_time))
        
    wind[:,:,0] = np.random.multivariate_normal(mean, cov, n)
    return wind 

def forward_euler(time, wind, wind_scale, g, l, m, theta0s, vel0s, dt, damp_coeff, linear):
    thetas = []
    theta0 = theta0s.copy()
    vel0 = vel0s.copy()

    step_size = int(time/dt)
    time_array = np.linspace(0, time, step_size) 
    for tstep in range(step_size):
        if linear: 
            a = (l * np.cos(theta0) * wind_scale * wind[:, tstep] - damp_coeff * vel0 - m*g*l * theta0) / (m*l**2) # wind force
            # a = (l * np.cos(theta0) * wind_scale * 5 - damp_coeff * vel0 - m*g*l * theta0) / (m*l**2) # constant wind force
            # a = (- damp_coeff * vel0 - m*g*l * theta0) / (m*l**2) # no wind force
        else: 
            # a = (l * np.cos(theta0) * wind_scale * wind[:, tstep]  - damp_coeff * vel0 - m*g*l * np.sin(theta0)) / (m*l**2) # wind force
            a = (wind_scale * wind[:, tstep]  - damp_coeff * vel0 - m*g*l * np.sin(theta0)) / (m*l**2) # wind force

            # a = (l * np.cos(theta0) * wind_scale * 5 - damp_coeff * vel0 - m*g*l * np.sin(theta0)) / (m*l**2) # constant wind force
            # a = (- damp_coeff * vel0 - m*g*l * np.sin(theta0)) / (m*l^2) # no wind force
        vel0 = vel0 + a * dt
        theta0 = theta0 + vel0 * dt
        thetas.append(theta0)
        
    thetas = np.array(thetas).transpose(1, 0, 2)

    return thetas

def generate_samples(n, time, dt, length_scale, variance, wind_scale, g, l, m, theta0, vel0, damp_coeff, linear=True):
    # sample wind
    wind = sample_wind(n, time, dt, length_scale, variance) 
    # use ode solver
    theta0s = np.full((n, 1), theta0)
    vel0s = np.full((n, 1), vel0)
    thetas = forward_euler(time, wind, wind_scale, g, l, m, theta0s, vel0s, dt, damp_coeff, linear)

    return thetas.astype(np.float64), wind.astype(np.float64)

def main():
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument('num_samples', type=int, default=100)
    parser.add_argument('--subsample_step_size', type=float, default=10)
    parser.add_argument('data_path', type=Path)

    # pendulum params
    parser.add_argument('--motion_type', type=str, default="nonlinear", choices=["linear", "nonlinear"])
    parser.add_argument('--pend_length', type=float, default=1)
    parser.add_argument('--pend_mass', type=float, default=1)
    parser.add_argument('--damp_coeff', type=float, default=0.5)
    parser.add_argument('--theta0', type=lambda x: np.radians(float(x)), default=0)
    parser.add_argument('--vel0', type=float, default=0)

    # sim params
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sim_time', type=int, default=10)
    parser.add_argument('--sim_dt', type=float, default=0.01)

    # latent force params
    parser.add_argument('--wind_length_scale', type=float, default=1)
    parser.add_argument('--wind_variance', type=float, default=1)
    parser.add_argument('--wind_scale', type=float, default=1)

    # video params
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--height', type=int, default=40)
    parser.add_argument('--pad', type=float, default=0.9) 
    parser.add_argument('--radius', type=float, default=3)    
    
    args = parser.parse_args()

    args.data_path.mkdir(exist_ok=True)
    
    thetas, winds = generate_samples(
        args.num_samples,
        args.sim_time,
        args.sim_dt,
        args.wind_length_scale,
        args.wind_variance,
        args.wind_scale,
        GRAVITY,
        args.pend_length,
        args.pend_mass,
        args.theta0,
        args.vel0,
        args.damp_coeff,
        args.motion_type=="linear"
    )

    # subsample 
    thetas = thetas[:,::args.subsample_step_size]
    winds = winds[:,::args.subsample_step_size]

    # generate video frames
    x, y = theta_to_pos(thetas, args.width, args.height, args.pend_length, args.pad)
    videos = generate_video_frames(x, y, args.width, args.height, args.radius)

    with open(f"{args.data_path}/{args.motion_type}.pkl", "wb") as f:
        pickle.dump({
            "thetas": thetas,
            "winds": winds,
            "frames": videos,
            "time": args.sim_time,
            "dt": args.sim_dt*args.subsample_step_size, 
        }, f) 
    

if __name__ == "__main__":
    main()    