# ============================================================
# PARTE 1
# Poner después de la función richardson()
# ============================================================

def trapezoid_step(t: np.array, velocity: np.array, ii: int):
    h = t[ii] - t[ii-1]

    integral = (h/2) * (
        velocity[ii-1] + velocity[ii]
    )

    return integral


def simpson_13_step(t: np.array, velocity: np.array, ii: int):
    h1 = t[ii-1] - t[ii-2]
    h2 = t[ii] - t[ii-1]

    if not np.isclose(h1, h2):
        return None

    h = h1

    integral = (h/3) * (
        velocity[ii-2]
        + 4*velocity[ii-1]
        + velocity[ii]
    )

    return integral


def reconstruct_trapezoid(
    t: np.array,
    velocity: np.array,
    initial_position: float
):
    position = [initial_position]

    for ii in range(1, len(t)):

        integral = trapezoid_step(
            t,
            velocity,
            ii
        )

        position.append(
            position[ii-1] + integral
        )

    return position


def reconstruct_simpson(
    t: np.array,
    velocity: np.array,
    initial_position: float
):
    position = [initial_position]

    for ii in range(1, len(t)):

        if ii == 1:

            integral = trapezoid_step(
                t,
                velocity,
                ii
            )

        else:

            integral = simpson_13_step(
                t,
                velocity,
                ii
            )

            if integral is None:

                integral = trapezoid_step(
                    t,
                    velocity,
                    ii
                )

        position.append(
            position[ii-1] + integral
        )

    return position


# ============================================================
# PARTE 2
# Poner después del bloque de cálculo de aceleración
# ============================================================

x0 = data["x"].iloc[0]
y0 = data["y"].iloc[0]

x_trap = reconstruct_trapezoid(
    data["t"],
    v_x_cn,
    x0
)

y_trap = reconstruct_trapezoid(
    data["t"],
    v_y_cn,
    y0
)

x_simp = reconstruct_simpson(
    data["t"],
    v_x_cn,
    x0
)

y_simp = reconstruct_simpson(
    data["t"],
    v_y_cn,
    y0
)


# ============================================================
# PARTE 3
# Poner después de la Parte 2
# ============================================================

error_x_trap_abs = absolute_error(
    np.array(data["x"]),
    np.array(x_trap)
)

error_y_trap_abs = absolute_error(
    np.array(data["y"]),
    np.array(y_trap)
)

error_x_simp_abs = absolute_error(
    np.array(data["x"]),
    np.array(x_simp)
)

error_y_simp_abs = absolute_error(
    np.array(data["y"]),
    np.array(y_simp)
)

error_x_trap_rel = relative_error(
    np.array(data["x"]),
    np.array(x_trap)
)

error_y_trap_rel = relative_error(
    np.array(data["y"]),
    np.array(y_trap)
)

error_x_simp_rel = relative_error(
    np.array(data["x"]),
    np.array(x_simp)
)

error_y_simp_rel = relative_error(
    np.array(data["y"]),
    np.array(y_simp)
)


# ============================================================
# PARTE 4
# Poner después de la función rel_error_plot()
# ============================================================

def trajectory_reconstruction_plot(
    x_original,
    y_original,
    x_trap,
    y_trap,
    x_simp,
    y_simp,
    folder_name="results"
):

    os.makedirs(folder_name, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(
        x_original,
        y_original,
        label="Trayectoria original"
    )

    plt.plot(
        x_trap,
        y_trap,
        label="Reconstruida - Trapecio"
    )

    plt.plot(
        x_simp,
        y_simp,
        label="Reconstruida - Simpson 1/3"
    )

    plt.xlabel("Position X")
    plt.ylabel("Position Y")

    plt.title(
        "Trayectoria original vs. reconstruida"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    file_path = os.path.join(
        folder_name,
        "trajectory_reconstruction.pdf"
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# PARTE 5
# Poner dentro de main(), después de las demás gráficas
# ============================================================

trajectory_reconstruction_plot(
    data["x"],
    data["y"],
    x_trap,
    y_trap,
    x_simp,
    y_simp
)
