import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse 

def read_data(file_name:str):
    points = pd.read_csv(f"data/{file_name}", sep='\\s+')
    return points 

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

def absolute_error(true:float, experimental:float):
    return np.abs(true - experimental)

def relative_error(true:float, experimental:float):
    return absolute_error(true, experimental)/true

def position_plot(time, x_pos, y_pos, folder_name='results'):
    os.makedirs(folder_name, exist_ok=True) # Create the folder if it doesn't exist

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax1.plot(time, x_pos, color='tab:blue', linewidth=2, label='x(t)')
    ax1.set_title('Position X vs Time')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Position X')
    ax1.grid(True)

    ax2.plot(time, y_pos, color='tab:orange', linewidth=2, label='y(t)')
    ax2.set_title('Position Y vs Time')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Position Y')
    ax2.grid(True)

    ax3.plot(x_pos, y_pos, color='tab:green', linewidth=2, label='y(x)')
    ax3.set_title('Position Y vs Position X')
    ax3.set_xlabel('Position X')
    ax3.set_ylabel('Position Y')
    ax3.grid(True)

    plt.tight_layout()
    file_path = os.path.join(folder_name, f"positions.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches='tight') # Save the plot as a PDF file
    plt.show()
    plt.close(fig) # Close the plot to free memory


def grouped_plot(mode, time, arrays_x, arrays_y, labels, folder_name='results'):
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
    plt.show()
    plt.close(fig)

def rel_error_plot(time, arrays, titles, labels, folder_name='results'):
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
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Numerical experiment')
    parser.add_argument('--data_file', type=str, help='Name of the file that has the data.')
    parser.add_argument('--no_plot', action='store_true', help='If set, the program will not generate plots.')
    args = parser.parse_args()

    data_file = args.data_file
    no_plot = args.no_plot

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


    position_plot(data["t"], data["x"], data["y"])

    lables = [None, "backward", "forward", "central", "central-Richardson"]
    vel_x = [data["vx"], v_x_bw, v_x_fw, v_x_cn, v_x_rc]
    vel_y = [data["vy"], v_y_bw, v_y_fw, v_y_cn, v_y_rc]
    grouped_plot("vel", data["t"], vel_x, vel_y, lables)

    acc_x = [data["vx"], a_x_bw, a_x_fw, a_x_cn, a_x_rc]
    acc_y = [data["vy"], a_y_bw, a_y_fw, a_y_cn, a_y_rc]
    grouped_plot("acc", data["t"], acc_x, acc_y, lables)

    rel_titles = ["Relative error of V_x", "Relative error of V_y", "Relative error of A_x", "Relative error of A_y"]
    rel_lables = ["backward", "forward", "central", "central-Richardson"]
    rel_errors = [rel_error_V_x, rel_error_V_y, rel_error_A_x, rel_error_A_y]
    rel_error_plot(data["t"], rel_errors, rel_titles, rel_lables)

    print(f"Relative error (Velocity-X backward)={np.mean(rel_error_v_x_bw)}")
    print(f"Relative error (Velocity-X forward)={np.mean(rel_error_v_x_fw)}")
    print(f"Relative error (Velocity-X central)={np.mean(rel_error_v_x_cn)}")
    print(f"Relative error (Velocity-X central-Richardson)={np.mean(rel_error_v_x_rc)}")   
    print("#"*30) 
    print(f"Relative error (Acceleration-X backward)={np.mean(rel_error_a_x_bw)}")
    print(f"Relative error (Acceleration-X forward)={np.mean(rel_error_a_x_fw)}")
    print(f"Relative error (Acceleration-X central)={np.mean(rel_error_a_x_cn)}")
    print(f"Relative error (Acceleration-X central-Richardson)={np.mean(rel_error_a_x_rc)}")


if __name__ == "__main__":
    main()