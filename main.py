import matplotlib
import os
import pandas as pd
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.interpolate import CubicSpline

from Vehicle_model import VehicleIntegrator, VehicleInput, VehicleState, DynamicBicycleModel
from Vehicle_controller import PurePursuit


# ==============================================================
#            TRAIETTORIE PER LA SIMULAZIONE
# ==============================================================

"""def generate_circular_path(radius: float, center: Tuple[float, float],
                           num_points: int = 150) -> List[Tuple[float, float]]:
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points)

    path = []
    for angle in angles:
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        path.append((x, y))

    return path"""


"""def generate_figure_eight_path(width: float, num_points: int = 200) -> List[Tuple[float, float]]:

    path = []
    t = np.linspace(0, 2 * np.pi, num_points)

    # Lemniscata di Gerono
    x = width * np.sin(t)
    y = width * np.sin(t) * np.cos(t)

    for i in range(num_points):
        path.append((x[i], y[i]))

    return path"""


"""def generate_straight_and_turn_path(straight_len: float, turn_radius: float,
                                    num_points: int = 150) -> List[Tuple[float, float]]:

    path = []

    # Parte 1: Linea retta
    num_straight = int(num_points * 0.6)
    for x in np.linspace(0, straight_len, num_straight):
        path.append((x, 0.0))

    # Parte 2: Curva a 90°
    num_turn = num_points - num_straight
    center_x = straight_len
    center_y = turn_radius

    for angle in np.linspace(-np.pi / 2, 0, num_turn):
        px = center_x + turn_radius * np.cos(angle)
        py = center_y + turn_radius * np.sin(angle)
        path.append((px, py))

    return path"""


def generate_random_spline_path(num_waypoints: int = 7,
                                x_range: Tuple[float, float] = (0, 10),
                                y_range: Tuple[float, float] = (-3, 3),
                                num_points: int = 300) -> List[Tuple[float, float]]:

    waypoints_x = np.linspace(x_range[0], x_range[1], num_waypoints)
    waypoints_y = np.random.uniform(y_range[0], y_range[1], num_waypoints)

    # Punto iniziale vincolato a (0, 0)
    waypoints_x[0] = 0
    waypoints_y[0] = 0

    # interpolazione tramite spline
    spline = CubicSpline(waypoints_x, waypoints_y)

    x_new = np.linspace(waypoints_x[0], waypoints_x[-1], num_points)
    y_new = spline(x_new)

    path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
    return path


# ==============================================================
#                    SIMULAZIONE
# ==============================================================

def run_simulation():
    dt = 0.01  # [s]
    T_sim = 20.0
    num_steps = int(T_sim / dt)

    d_constant = 0.25  # duty cycle (fisso)

    # ===== TRAIETTORIA DI RIFERIMENTO =====
    #radius = 2.0
    #center = (2.0, 0.0)


    #reference_path = generate_circular_path(radius, center, num_points=150)
    #reference_path = generate_figure_eight_path(width=4.0, num_points=200)
    # reference_path = generate_straight_and_turn_path(straight_len=3.0, turn_radius=1.5, num_points=200)
    reference_path = generate_random_spline_path(num_waypoints=8, num_points=300)

    print(f"Numero waypoints: {len(reference_path)}")

    # Modello + Controller
    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    controller = PurePursuit(wheelbase=0.062)

    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    # Storage
    trajectory_x = [state.X]
    trajectory_y = [state.Y]
    time_history = [0.0]
    vx_history = [state.vx]
    vy_history = [state.vy]
    omega_history = [state.omega]

    print(f"\nInizio simulazione: {num_steps} step, dt={dt}s")

    for step in range(num_steps):

        current_time = step * dt

        delta = controller.pure_pursuit(state, reference_path)

        control_input = VehicleInput(d=d_constant, delta=delta)
        control_input.saturate()



        state = integrator.Eulero(state, control_input)

        trajectory_x.append(state.X)
        trajectory_y.append(state.Y)
        time_history.append(current_time + dt)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        omega_history.append(state.omega)

        if step % int(2 / dt) == 0:
            print(f"t={current_time:.1f}s: pos=({state.X:.3f}, {state.Y:.3f}), "
                  f"vx={state.vx:.3f}, vy={state.vy:.4f}")

    print("\nSimulazione completata!")

    # ==============================================================
    #                STATISTICHE
    # ==============================================================

    errors = []
    for x, y in zip(trajectory_x, trajectory_y):
        min_dist = min(np.sqrt((x - rx) ** 2 + (y - ry) ** 2) for rx, ry in reference_path)
        errors.append(min_dist)

    print("\n=========== STATISTICHE FINE SIMULAZIONE ===========")
    print(f"Errore medio: {np.mean(errors):.4f} m")
    print(f"Errore massimo: {np.max(errors):.4f} m")
    print(f"RMSE errore: {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m")
    print("====================================================")

    # ==============================================================
    #                PLOT
    # ==============================================================

    # PLOT TRAIETTORIA DI RIFERIMENTO
    ref_x = [p[0] for p in reference_path]
    ref_y = [p[1] for p in reference_path]

    plt.figure(figsize=(8, 6))
    plt.plot(ref_x, ref_y, 'r-', linewidth=2)
    plt.title("Traiettoria di riferimento (solo percorso)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Traiettoria XY con percorso seguito
    plt.figure(figsize=(10, 8))
    plt.plot(ref_x, ref_y, 'r--', linewidth=2, label='Traiettoria riferimento')
    plt.plot(trajectory_x, trajectory_y, 'b-', linewidth=1.8, label='Traiettoria seguita')
    plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=8, label='Start')
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'ms', markersize=8, label='End')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Traiettoria con Pure Pursuit')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Stati nel tempo
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(time_history, vx_history)
    axes[0].set_ylabel('vx [m/s]')
    axes[0].grid(True)

    axes[1].plot(time_history, vy_history)
    axes[1].set_ylabel('vy [m/s]')
    axes[1].grid(True)

    axes[2].plot(time_history, omega_history)
    axes[2].set_ylabel('ω [rad/s]')
    axes[2].grid(True)

    axes[3].plot(time_history, errors)
    axes[3].set_ylabel('Errore [m]')
    axes[3].set_xlabel('Tempo [s]')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

def run_vx_test():
        dt = 0.01
        T_sim = 10.0
        num_steps = int(T_sim / dt)

        model = DynamicBicycleModel()
        integrator = VehicleIntegrator(model, dt)

        state = VehicleState(
            X=0.0, Y=0.0,
            phi=0.0,
            vx=0.0,
            vy=0.0,
            omega=0.0
        )

        vx_des = 0.5
        kp = 85

        time_history = []
        vx_history = []
        error_history = []
        d_history = []

        print("\n=== INIZIO TEST LONGITUDINALE SU v_x ===")

        for step in range(num_steps):
            t = step * dt

            error_vx = vx_des - state.vx

            d_cmd = kp * error_vx
            d_cmd = np.clip(d_cmd, 0.0, 1.0)

            delta_cmd = 0.0

            control_input = VehicleInput(d=d_cmd, delta=delta_cmd)
            state = integrator.Eulero(state, control_input)

            time_history.append(t)
            vx_history.append(state.vx)
            error_history.append(error_vx)
            d_history.append(d_cmd)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(10, 8))

        ax[0].plot(time_history, vx_history, label="v_x(t)")
        ax[0].axhline(vx_des, color='r', linestyle='--', label="v_x_des")
        ax[0].set_ylabel("v_x [m/s]")
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(time_history, error_history)
        ax[1].set_ylabel("Errore v_x")
        ax[1].grid()

        ax[2].plot(time_history, d_history)
        ax[2].set_ylabel("duty cycle d(t)")
        ax[2].set_xlabel("Tempo [s]")
        ax[2].grid()

        plt.tight_layout()

        # Crea cartella principale se non esiste
        base_dir = "misurazioni_Kp_incrementale"
        os.makedirs(base_dir, exist_ok=True)

        # Cartella dedicata al valore di kp
        folder_name = f"kp_{kp:.3f}".replace(".", "_")
        save_dir = os.path.join(base_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nSalvataggio risultati in: {save_dir}\n")

        # ---- SALVA I DATI NUMERICI ----
        df = pd.DataFrame({
            "time": time_history,
            "vx": vx_history,
            "error_vx": error_history,
            "duty_cycle": d_history
        })

        df.to_csv(os.path.join(save_dir, "dati.csv"), index=False)
        print("→ dati.csv salvato")

        # ---- SALVA I PLOT ----
        fig.savefig(os.path.join(save_dir, "grafici_vx_error_duty.png"))
        print("→ grafici_vx_error_duty.png salvato")

        # ---- SALVA LOG ----
        with open(os.path.join(save_dir, "log.txt"), "w") as f:
            f.write(f"Test controllo v_x\n")
            f.write(f"kp = {kp}\n")
            f.write(f"vx_des = {vx_des}\n")
            f.write(f"durata sim = {T_sim}s\n")
            f.write(f"dt = {dt}\n")
            f.write(f"vx finale = {vx_history[-1]}\n")

        print("→ log.txt salvato")
        plt.show()


if __name__ == "__main__":
    run_vx_test()
   # run_simulation()

