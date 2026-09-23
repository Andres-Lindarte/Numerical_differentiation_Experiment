import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_name:str):
    points = pd.read_csv(f"data/{file_name}", sep='\\s+')
    return points 

def backward(x:np.array, t:np.array, ii):
    h = t[ii] - t[ii-1] #Careful with ii
    return (x[ii]-(x[ii-1]))/h
    
def forward(x:np.array, t:np.array, ii):
    h = t[ii] - t[ii-1] #Careful with ii
    return ((x[ii+1])-x[ii])/h

def central(x:np.array, t:np.array, ii):
    h = t[ii] - t[ii-1] #Careful with ii
    return ((x[ii+1])-x[ii-1])/(2*h)

def richardson():
    pass

def plot(time, array1, array2, array3, array4):
    plt.figure(figsize=(10, 6))

    # Trazar cada una de las series
    plt.plot(time, array1, label="etiquetas[0]", color="#1f77b4", linewidth=2)
    plt.plot(time, array2, label="etiquetas[1]", color="#ff7f0e", linewidth=2)
    plt.plot(time, array3, label="etiquetas[2]", color="#2ca02c", linewidth=2)
    plt.plot(time, array4, label="etiquetas[3]", color="#d62728", linewidth=2)

    # Configuración estéticas de la gráfica
    plt.title("titulo", fontsize=14, fontweight="bold")
    plt.xlabel("eje_x", fontsize=12)
    plt.ylabel("eje_y", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    plt.show()
    

def main():
    data = read_data("raw_data.txt")
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
    
    for ii in range(0, 22):
        if ii == 0:
            v_x_bw.append(0)
            v_y_bw.append(0)
            v_x_fw.append(0)
            v_y_fw.append(0)
            #v_x_fw.append(forward(data["x"], data["t"], ii))
            #v_y_fw.append(forward(data["y"], data["t"], ii))
        elif ii != 0 and ii != 21 :
            v_x_bw.append(backward(data["x"], data["t"], ii))
            v_y_bw.append(backward(data["y"], data["t"], ii))
            v_x_fw.append(forward(data["x"], data["t"], ii))
            v_y_fw.append(forward(data["y"], data["t"], ii))
        elif ii == 21:
            v_x_bw.append(backward(data["x"], data["t"], ii))
            v_y_bw.append(backward(data["y"], data["t"], ii))
            v_x_fw.append(0)
            v_y_fw.append(0)

    plot(data["t"], v_x_bw, v_x_fw, v_y_bw, v_y_fw)

if __name__ == "__main__":
    main()