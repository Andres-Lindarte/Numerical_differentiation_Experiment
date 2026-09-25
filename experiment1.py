import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_name:str):
    points = pd.read_csv(f"data/{file_name}", sep=r'\\s+')
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

def central_2(x:np.array, t:np.array, ii):
    h = (t[ii] - t[ii-1])/2
    return ((x[ii+1])-x[ii-1])/(2*h)

def richardson(x:np.array, t:np.array, ii):
    return (central_2(x,t,ii))+(1/3)*(central_2(x,t,ii)-central(x,t,ii))


def plot(time, array1, array2, array3, array4, array5, array6, array7, array8, labels, title):
    plt.figure(figsize=(10, 6))

    all_arrays = [array1, array2, array3, array4, array5, array6, array7, array8]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for i in range(len(all_arrays)):
        if labels[i] is not None:
            plt.plot(time, all_arrays[i], label=labels[i], color=colors[i], linewidth=2)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Tiempo (t)", fontsize=12)
    plt.ylabel("Valor de la Derivada", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    plt.show()


def main():
    data = read_data("raw_data.txt")

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
            v_x_cn.append(0)
            v_y_cn.append(0)
            v_x_rc.append(0)
            v_y_rc.append(0)
        elif ii != 0 and ii != 21 :
            v_x_bw.append(backward(data["x"], data["t"], ii))
            v_y_bw.append(backward(data["y"], data["t"], ii))
            v_x_fw.append(forward(data["x"], data["t"], ii))
            v_y_fw.append(forward(data["y"], data["t"], ii))
            v_x_cn.append(central(data["x"], data["t"], ii))
            v_y_cn.append(central(data["y"], data["t"], ii))
            v_x_rc.append(richardson(data["x"], data["t"], ii))
            v_y_rc.append(richardson(data["y"], data["t"], ii))
        elif ii == 21:
            v_x_bw.append(backward(data["x"], data["t"], ii))
            v_y_bw.append(backward(data["y"], data["t"], ii))
            v_x_fw.append(0)
            v_y_fw.append(0)
            v_x_cn.append(0)
            v_y_cn.append(0)
            v_x_rc.append(0)
            v_y_rc.append(0)


    plot_labels_x = [
        "Derivada Backward X",
        "Derivada Forward X",
        "Derivada Central X",
        "Derivada Richardson X",
        None,
        None,
        None,
        None
    ]

    plot(data["t"], v_x_bw, v_x_fw, v_x_cn, v_x_rc, np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), plot_labels_x, "Derivadas de la Coordenada X")


    plot_labels_y = [
        "Derivada Backward Y",
        "Derivada Forward Y",
        "Derivada Central Y",
        "Derivada Richardson Y",
        None,
        None,
        None,
        None
    ]
    # Graficar las derivadas de Y
    plot(data["t"], v_y_bw, v_y_fw, v_y_cn, v_y_rc, np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), plot_labels_y, "Derivadas de la Coordenada Y")

    #Segundas Derivadas

    a_x_bw = []
    a_x_fw = []
    a_x_cn = []
    a_x_rc = []

    a_y_bw = []
    a_y_fw = []
    a_y_cn = []
    a_y_rc = []

    for ii in range(0,22):
        if ii == 0:
            a_x_bw.append(0)
            a_y_bw.append(0)
            a_x_fw.append(0)
            a_y_fw.append(0)
            a_x_cn.append(0)
            a_y_cn.append(0)
            a_x_rc.append(0)
            a_y_rc.append(0)
        elif ii != 0 and ii != 21 :
            a_x_bw.append(backward(v_x_bw, data["t"], ii))
            a_y_bw.append(backward(v_y_bw, data["t"], ii))
            a_x_fw.append(forward(v_x_fw, data["t"], ii))
            a_y_fw.append(forward(v_y_fw, data["t"], ii))
            a_x_cn.append(central(v_x_cn, data["t"], ii))
            a_y_cn.append(central(v_y_cn, data["t"], ii))
            a_x_rc.append(richardson(v_x_rc, data["t"], ii))
            a_y_rc.append(richardson(v_y_rc, data["t"], ii))
        elif ii == 21:
            a_x_bw.append(backward(v_x_bw, data["t"], ii))
            a_y_bw.append(backward(v_y_bw, data["t"], ii))
            a_x_fw.append(0)
            a_y_fw.append(0)
            a_x_cn.append(0)
            a_y_cn.append(0)
            a_x_rc.append(0)
            a_y_rc.append(0)

    acceleration_labels_x = [
        "Aceleración Backward X","Aceleración Forward X","Aceleración Central X","Aceleración Richardson X",None,None,None,None
    ]
    plot(data["t"], a_x_bw, a_x_fw, a_x_cn, a_x_rc, np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), acceleration_labels_x, "Aceleraciones de la Coordenada X")

    acceleration_labels_y = [
        "Aceleración Backward Y","Aceleración Forward Y","Aceleración Central Y","Aceleración Richardson Y",None,None,None,None
    ]
    plot(data["t"], a_y_bw, a_y_fw, a_y_cn, a_y_rc, np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), np.zeros_like(data["t"]), acceleration_labels_y, "Aceleraciones de la Coordenada Y")

    #Diferenciación

    def trapezoid(x:np.array, y:np.array):
      n = len(x)
      h = x[1] - x[0]
      result = 0

      for ii in range(0, n-1):
        result += (h/2)*(y[ii] + y[ii+1])

      return result

if __name__ == "__main__":
    main()

