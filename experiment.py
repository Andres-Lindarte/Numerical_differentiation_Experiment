import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse 

# --- Read data file ---
def read_data(file_name:str):
    points = pd.read_csv(f"data/{file_name}", sep='\\s+')
    return points 

# --- Numerical differentiation methods ---
def backward(x:np.array, t:np.array, ii):
    h = t[ii] - t[ii-1] #Careful with ii
    return (x[ii]-(x[ii-1]))/h
    
def forward(x:np.array, t:np.array, ii):
    h = t[ii+1] - t[ii] #Careful with ii
    return ((x[ii+1])-x[ii])/h

def central(x:np.array, t:np.array, ii:int, h_factor=1):
    h = (t[ii] - t[ii-1])/h_factor #Careful with ii
    return ((x[ii+1])-x[ii-1])/(2*h)

def richardson(x:np.array, t:np.array, ii:int):
    return (central(x,t,ii,2))+(1/3)*(central(x,t,ii,2)-central(x,t,ii))

# --- Gravity estimation ---
def estimate_gravity(method_name: str, acceleration: np.array, g_theoretical=9.81):
    
    flight_a_y = np.array(acceleration)[2:-2]
    g_exp = np.abs(np.mean(flight_a_y))
    
    err_pct = relative_error(g_theoretical, g_exp) * 100
    
    print("\n" + "="*50)
    print(f" Gravity stimation ({method_name}) ")
    print("="*50)
    print(f"Experimental g : {g_exp:.4f} m/s²")
    print(f"Relative error : {err_pct:.2f}%")
    print("="*50 + "\n")

# --- Error calculations ---
def absolute_error(true:float, experimental:float):
    return np.abs(true - experimental)

def relative_error(true, experimental):
    true_arr = np.array(true)
    exp_arr = np.array(experimental)
    denominator = np.maximum(np.abs(true_arr), 1e-2)
    return absolute_error(true_arr, exp_arr) / denominator

# --- Numerical integration methods ---
def trapezoid_step(time: np.array, velocity: np.array, ii: int):
    h = time[ii] - time[ii-1]
    integral = (h/2) * (velocity[ii-1] + velocity[ii])
    return integral

def reconstruct_trapezoid(time: np.array, velocity: np.array, initial_position: float):
    position = [initial_position]
    for ii in range(1, len(time)):
        integral = trapezoid_step(time, velocity, ii)
        position.append(position[ii-1] + integral)
    return position

def simpson_13_step(time: np.array, velocity: np.array, ii: int):
    h1 = time[ii-1] - time[ii-2]
    h2 = time[ii] - time[ii-1]
    if not np.isclose(h1, h2):
        return None
    h = h1
    integral = (h/3) * (velocity[ii-2] + 4*velocity[ii-1] + velocity[ii])
    return integral

def reconstruct_simpson(time: np.ndarray, velocity: np.ndarray, initial_position: float):
    position = [initial_position]
    for ii in range(1, len(time)):
        if ii % 2 == 0:
            integral = simpson_13_step(time, velocity, ii)
            if integral is not None:
                position.append(position[ii-2] + integral)
            else:
                position.append(position[ii-1] + trapezoid_step(time, velocity, ii))
        else:
            position.append(position[ii-1] + trapezoid_step(time, velocity, ii))
    return position

# --- Plotting functions ---
def position_plot(show: bool, time: np.ndarray, x_pos: np.ndarray, y_pos: np.ndarray, x_real: np.ndarray=None, y_real: np.ndarray=None, method_name: str=None, folder_name: str='results'):
    os.makedirs(folder_name, exist_ok=True) # Create the folder if it doesn't exist

    has_real = (x_real is not None) and (y_real is not None) and (method_name is not None)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    label_x = f'Reconstructed {method_name}' if has_real else 'x(t)'
    ax1.plot(time, x_pos, color='tab:blue', linewidth=2, label=label_x)
    if has_real:
        ax1.plot(time, x_real, color='black', linestyle='--', linewidth=1.5, label='Real')
        ax1.legend()
    ax1.set_title('Position X vs Time')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Position X')
    ax1.grid(True)

    label_y = f'Reconstructed {method_name}' if has_real else 'y(t)'
    ax2.plot(time, y_pos, color='tab:orange', linewidth=2, label=label_y)
    if has_real:
        ax2.plot(time, y_real, color='black', linestyle='--', linewidth=1.5, label='Real')
        ax2.legend()
    ax2.set_title('Position Y vs Time')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Position Y')
    ax2.grid(True)

    label_xy = f'Reconstructed {method_name}' if has_real else 'y(x)'
    ax3.plot(x_pos, y_pos, color='tab:green', linewidth=2, label=label_xy)
    if has_real:
        ax3.plot(x_real, y_real, color='black', linestyle='--', linewidth=1.5, label='Real')
        ax3.legend()
    ax3.set_title('Position Y vs Position X')
    ax3.set_xlabel('Position X')
    ax3.set_ylabel('Position Y')
    ax3.grid(True)

    plt.tight_layout()
    file_path = os.path.join(folder_name, "positions.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches='tight') # Save the plot as a PDF file
    plt.show() if show else None
    plt.close(fig) # Close the plot to free memory


def grouped_plot(show: bool, mode: str, time: np.ndarray, arrays_x: list, arrays_y: list, labels: list, folder_name: str = 'results'):
    os.makedirs(folder_name, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))

    plots_config = [
        (ax1, arrays_x, "x"),
        (ax2, arrays_y, "y")]

    for ax, array_data, axis in plots_config:
        for ii, name in enumerate(labels):
            if ii == 0:
                lbl = f'Tracker {mode}_{axis}(t)'
            else:
                lbl = f'{mode}_{axis}_{name}'
            
            ax.plot(time, array_data[ii], label=lbl)

        ax.set_title(f'{mode.upper()} {axis.upper()}', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{axis.upper()} component')
        ax.grid(True)
        ax.legend() 

    plt.tight_layout()

    file_path = os.path.join(folder_name, f"{mode}.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show() if show else None
    plt.close(fig)

def rel_error_plot(show: bool, time: np.ndarray, arrays: list, titles: list, labels: list, folder_name: str = 'results'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for ii, ax in enumerate(axes.flat):
        current_group = arrays[ii]
        
        for jj, plot in enumerate(current_group):
            if ii == 0:
                kk = jj
            else:
                kk = jj +4
            ax.plot(time, plot, label=labels[jj], linewidth=1.5)

        ax.set_title(titles[ii], fontsize=10, fontweight='bold')
        ax.set_ylabel('Relative error', fontsize=9)
        ax.grid(True)
        ax.legend(fontsize=8, loc='best')
        
        if ii >= 2:
            ax.set_xlabel('Time (t)', fontsize=9)

    plt.tight_layout()

    file_path = os.path.join(folder_name, f"relative_error.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show() if show else None
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Numerical experiment')
    parser.add_argument('--data_file', type=str, default="raw_data", help='Name of the file that has the data.')
    parser.add_argument('--output_file', type=str, default="results.txt", help='Name of the file to save the results.')
    parser.add_argument('--no_plot', action='store_true', help='If set, the plots will not be shown, but saved.')
    parser.add_argument('--gravity_estimation', action='store_true', help='If set, the gravity estimation will be performed.')
    args = parser.parse_args()

    data_file = args.data_file
    out_file = args.output_file
    plot = not args.no_plot
    gravity_estimation = args.gravity_estimation

    data = read_data(f"{data_file}.txt")
    #print(data.head())
    #print(len(data["t"]))

    v_x_bw = []
    v_x_fw = []
    v_x_cn = []
    v_x_rc = []

    v_y_bw = []
    v_y_fw = []
    v_y_cn = []
    v_y_rc = []    

    a_x_bw = []
    a_x_fw = []
    a_x_cn = []
    a_x_rc = []

    a_y_bw = []
    a_y_fw = []
    a_y_cn = []
    a_y_rc = []

    #Velocity
    for ii in range(0, len(data["t"])):
        if ii == 0:
            v_x_bw.append(0)
            v_y_bw.append(0)
            v_x_fw.append(forward(data["x"], data["t"], ii))
            v_y_fw.append(forward(data["y"], data["t"], ii))
            v_x_cn.append(0)
            v_y_cn.append(0)
            v_x_rc.append(0)
            v_y_rc.append(0)
        elif ii != 0 and ii != len(data["t"])-1:
            v_x_bw.append(backward(data["x"], data["t"], ii))
            v_y_bw.append(backward(data["y"], data["t"], ii))
            v_x_fw.append(forward(data["x"], data["t"], ii))
            v_y_fw.append(forward(data["y"], data["t"], ii))
            v_x_cn.append(central(data["x"], data["t"], ii))
            v_y_cn.append(central(data["y"], data["t"], ii))
            v_x_rc.append(richardson(data["x"], data["t"], ii))
            v_y_rc.append(richardson(data["y"], data["t"], ii))
        elif ii == len(data["t"])-1:
            v_x_bw.append(backward(data["x"], data["t"], ii))
            v_y_bw.append(backward(data["y"], data["t"], ii))
            v_x_fw.append(0)
            v_y_fw.append(0)
            v_x_cn.append(0)
            v_y_cn.append(0)
            v_x_rc.append(0)
            v_y_rc.append(0)
    
    rel_error_v_x_bw = relative_error(data["vx"], v_x_bw)
    rel_error_v_y_bw = relative_error(data["vy"], v_y_bw)
    rel_error_v_x_fw = relative_error(data["vx"], v_x_fw)
    rel_error_v_y_fw = relative_error(data["vy"], v_y_fw)
    rel_error_v_x_cn = relative_error(data["vx"], v_x_cn)
    rel_error_v_y_cn = relative_error(data["vy"], v_y_cn)
    rel_error_v_x_rc = relative_error(data["vx"], v_x_rc)
    rel_error_v_y_rc = relative_error(data["vy"], v_y_rc)

    rel_error_V_x = [rel_error_v_x_bw, rel_error_v_x_fw, rel_error_v_x_cn, rel_error_v_x_rc]
    rel_error_V_y = [rel_error_v_y_bw, rel_error_v_y_fw, rel_error_v_y_cn, rel_error_v_y_rc]
    #print(rel_error_V_x)
    #print(rel_error_V_y)

    #Acceleration
    for ii in range(0, len(data["t"])):
        if ii == 0:
            a_x_bw.append(0)
            a_y_bw.append(0)
            a_x_fw.append(forward(v_x_fw, data["t"], ii))
            a_y_fw.append(forward(v_y_fw, data["t"], ii))
            a_x_cn.append(0)
            a_y_cn.append(0)
            a_x_rc.append(0)
            a_y_rc.append(0)
        elif ii != 0 and ii != len(data["t"])-1:
            a_x_bw.append(backward(v_x_bw, data["t"], ii))
            a_y_bw.append(backward(v_y_bw, data["t"], ii))
            a_x_fw.append(forward(v_x_fw, data["t"], ii))
            a_y_fw.append(forward(v_y_fw, data["t"], ii))
            a_x_cn.append(central(v_x_cn, data["t"], ii))
            a_y_cn.append(central(v_y_cn, data["t"], ii))
            a_x_rc.append(richardson(v_x_rc, data["t"], ii))
            a_y_rc.append(richardson(v_y_rc, data["t"], ii))
        elif ii == len(data["t"])-1:
            a_x_bw.append(backward(v_x_bw, data["t"], ii))
            a_y_bw.append(backward(v_y_bw, data["t"], ii))
            a_x_fw.append(0)
            a_y_fw.append(0)
            a_x_cn.append(0)
            a_y_cn.append(0)
            a_x_rc.append(0)
            a_y_rc.append(0)

    rel_error_a_x_bw = relative_error(data["ax"], a_x_bw)
    rel_error_a_y_bw = relative_error(data["ay"], a_y_bw)
    rel_error_a_x_fw = relative_error(data["ax"], a_x_fw)
    rel_error_a_y_fw = relative_error(data["ay"], a_y_fw)
    rel_error_a_x_cn = relative_error(data["ax"], a_x_cn)
    rel_error_a_y_cn = relative_error(data["ay"], a_y_cn)
    rel_error_a_x_rc = relative_error(data["ax"], a_x_rc)
    rel_error_a_y_rc = relative_error(data["ay"], a_y_rc)

    rel_error_A_x = [rel_error_a_x_bw, rel_error_a_x_fw, rel_error_a_x_cn, rel_error_a_x_rc]
    rel_error_A_y = [rel_error_a_y_bw, rel_error_a_y_fw, rel_error_a_y_cn, rel_error_a_y_rc]
    #print(rel_error_A_x)
    #print(rel_error_A_y)

    if gravity_estimation:
        fout.write(estimate_gravity("central", a_y_cn))

    position_plot(plot, data["t"], data["x"], data["y"])

    lables = [None, "backward", "forward", "central", "central-Richardson"]
    vel_x = [data["vx"], v_x_bw, v_x_fw, v_x_cn, v_x_rc]
    vel_y = [data["vy"], v_y_bw, v_y_fw, v_y_cn, v_y_rc]
    grouped_plot(plot, "vel", data["t"], vel_x, vel_y, lables)

    acc_x = [data["vx"], a_x_bw, a_x_fw, a_x_cn, a_x_rc]
    acc_y = [data["vy"], a_y_bw, a_y_fw, a_y_cn, a_y_rc]
    grouped_plot(plot, "acc", data["t"], acc_x, acc_y, lables)

    rel_titles = ["Relative error of V_x", "Relative error of V_y", "Relative error of A_x", "Relative error of A_y"]
    rel_lables = ["backward", "forward", "central", "central-Richardson"]
    rel_errors = [rel_error_V_x, rel_error_V_y, rel_error_A_x, rel_error_A_y]
    rel_error_plot(plot, data["t"], rel_errors, rel_titles, rel_lables)

    fout = open(f"results/{out_file}", "w") # Open a file to write the results
    fout.write(f"--- Mean relative error ---\n")
    fout.write(f"(Velocity-X backward)={np.mean(rel_error_v_x_bw[1:-1]):.4f}\n")
    fout.write(f"(Velocity-X forward)={np.mean(rel_error_v_x_fw[1:-1]):.4f}\n")
    fout.write(f"(Velocity-X central)={np.mean(rel_error_v_x_cn[1:-1]):.4f}\n")
    fout.write(f"(Velocity-X central-Richardson)={np.mean(rel_error_v_x_rc[1:-1]):.4f}\n")
    fout.write("#"*30)
    fout.write("\n")
    fout.write(f"(Acceleration-X backward)={np.mean(rel_error_a_x_bw[2:-2]):.4f}\n")
    fout.write(f"(Acceleration-X forward)={np.mean(rel_error_a_x_fw[2:-2]):.4f}\n")
    fout.write(f"(Acceleration-X central)={np.mean(rel_error_a_x_cn[2:-2]):.4f}\n")
    fout.write(f"(Acceleration-X central-Richardson)={np.mean(rel_error_a_x_rc[2:-2]):.4f}\n")

    #Integration: position reconstruction
    x0 = data["x"].iloc[0]
    y0 = data["y"].iloc[0]

    x_trap = reconstruct_trapezoid(data["t"], data["vx"], x0)
    y_trap = reconstruct_trapezoid(data["t"], data["vy"], y0)
    x_simp = reconstruct_simpson(data["t"], data["vx"], x0)
    y_simp = reconstruct_simpson(data["t"], data["vy"], y0)

    abs_error_x_trap = absolute_error(np.array(data["x"]), np.array(x_trap))
    abs_error_y_trap = absolute_error(np.array(data["y"]), np.array(y_trap))
    abs_error_x_simp = absolute_error(np.array(data["x"]), np.array(x_simp))
    abs_error_y_simp = absolute_error(np.array(data["y"]), np.array(y_simp))

    rel_error_x_trap = relative_error(np.array(data["x"]), np.array(x_trap))
    rel_error_y_trap = relative_error(np.array(data["y"]), np.array(y_trap))
    rel_error_x_simp = relative_error(np.array(data["x"]), np.array(x_simp))
    rel_error_y_simp = relative_error(np.array(data["y"]), np.array(y_simp))

    position_plot(plot, data["t"], x_trap, y_trap, x_real=data["x"], y_real=data["y"], method_name="Trapezoid")
    position_plot(plot, data["t"], x_simp, y_simp, x_real=data["x"], y_real=data["y"], method_name="Simpson")

    fout.write(f"\n--- Mean relative error of position reconstruction ---\n")
    fout.write(f"(Position-X Trapezoid)={np.mean(rel_error_x_trap[2:-2]):.4f}\n")
    fout.write(f"(Position-Y Trapezoid)={np.mean(rel_error_y_trap[2:-2]):.4f}\n")
    fout.write(f"(Position-X Simpson)={np.mean(rel_error_x_simp[2:-2]):.4f}\n")
    fout.write(f"(Position-Y Simpson)={np.mean(rel_error_y_simp[2:-2]):.4f}\n")

    fout.close() # Close the file after writing the results
    print(f"Results saved in 'results/{out_file}'")

if __name__ == "__main__":
    main()