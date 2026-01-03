import matplotlib
import os
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.interpolate import splev, splprep
from Veicolo.Vehicle_model import VehicleIntegrator, VehicleInput, VehicleState, DynamicBicycleModel
from Veicolo.Vehicle_controller import PurePursuit
from vxTest import run_vx_test
import datetime


# ==============================================================
#            TRAIETTORIE PER LA SIMULAZIONE
# ==============================================================

def generate_circular_path(radius: float, center: Tuple[float, float],
                           num_points: int = 150) -> List[Tuple[float, float]]:
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points) #si generano n valori equidistanti

    path = []
    for angle in angles:
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        path.append((x, y))

    return path


def generate_figure_eight_path(width: float, num_points: int = 200) -> List[Tuple[float, float]]:

    path = []
    t = np.linspace(0, 2 * np.pi, num_points)


    x = width * np.sin(t)
    y = width * np.sin(t) * np.cos(t)

    for i in range(num_points):
        path.append((x[i], y[i]))

    return path


def generate_straight_and_turn_path(straight_len: float, turn_radius: float,
                                    num_points: int = 150) -> List[Tuple[float, float]]:

    path = []

    num_straight = int(num_points * 0.6) #retta = 60 % punti
    for x in np.linspace(0, straight_len, num_straight):
        path.append((x, 0.0))

    num_turn = num_points - num_straight #i punti rimanenti dopo la linea retta
    center_x = straight_len
    center_y = turn_radius

    for angle in np.linspace(-np.pi / 2, 0, num_turn):
        px = center_x + turn_radius * np.cos(angle)
        py = center_y + turn_radius * np.sin(angle)
        path.append((px, py))

    return path
"""

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
    """


def generate_circuite_path(waypoints: List[Tuple[float, float]],
                           num_points: int = 300,
                           smoothness: int = 3) -> List[Tuple[float, float]]: #smoothness = grado della spline 3-> cubica

    waypoints_array = np.array(waypoints) #convertiamo in array
    x = waypoints_array[:, 0]
    y = waypoints_array[:, 1]

    tck, u = splprep([x, y], s=0, k=smoothness, per=1) #per = 1 chiude la curva. tck contiene il vettore dei nodi e i coefficienti della spline
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck) #rende la spline più liscia valutando i nuovi punti

    path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
    return path


def simulazione_traiettoria (ref_x, ref_y, trajectory_x, trajectory_y, dt, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot (ref_x, ref_y, 'r--',linewidth=2, label='Traiettoria Riferimento')

    all_x = ref_x + trajectory_x
    all_y = ref_y + trajectory_y
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Animazione traiettoria veicolo")
    ax.grid(True)
    ax.axis("equal")

    traj_line, = ax.plot([], [], 'b-', linewidth=2, label="Traiettoria seguita")


    vehicle_point, = ax.plot([], [], 'go', markersize=8, label="Veicolo")

    ax.legend()

    def init():
        traj_line.set_data([], [])
        vehicle_point.set_data([], [])
        return traj_line, vehicle_point

    def update(frame):

        x_traj = trajectory_x[:frame + 1]
        y_traj = trajectory_y[:frame + 1]
        traj_line.set_data(x_traj, y_traj)
        vehicle_point.set_data([trajectory_x[frame]], [trajectory_y[frame]])

        return traj_line, vehicle_point


    num_points = len(trajectory_x)

    step_skip = 20
    frames = range(0, num_points, step_skip)

    interval_ms = dt * step_skip * 1000

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=interval_ms,
        repeat=False
    )

    #writer = FFMpegWriter(fps=int(1.0 / (dt * step_skip)))
   # anim.save(output_path, writer=writer)

    plt.show()


    #plt.close(fig)







# ==============================================================
#                    SIMULAZIONE
# ==============================================================

def run_simulation():
    dt = 0.001  # passo di integrazione
    T_sim = 40.0 #durata simulazione
    num_steps = int(T_sim / dt) #numero di iterazioni del ciclo

    d_constant = 0.3  # duty cycle costante

    track_waypoints = [
        (0.0, 0.0),
        (0.5, 1.2),
        (1.2, 2.2),

        (2.5, 3.0),
        (4.5, 3.2),
        (6.5, 3.2),
        (8.2, 3.0),

        (9.0, 2.2),
        (9.2, 1.2),

        (8.5, 0.3),
        (6.8, -0.3),
        (4.7, -0.5),

        (3.5, -0.8),
        (2.3, -1.4),

        (0.5, -1.6),
        (-1.2, -1.4),
        (-2.0, -0.8),

        (-2.2, 0.0),
        (-1.5, 0.8),
        (-0.5, 1.0),


        (1.1, -0.6),
        (2.1, 0.0),


    ]

    # ===== TRAIETTORIA DI RIFERIMENTO =====
    radius = 2.0
    center = (2.0, 0.0)

    reference_path = generate_circuite_path(track_waypoints, num_points=300)
    trajectory_name = "Pista_Racing"
    #reference_path = generate_circular_path(radius, center, num_points=150)
    #trajectory_name = "Traiettoria_Circolare"
    #reference_path = generate_figure_eight_path(width=4.0, num_points=200)
    #trajectory_name = "Traiettoria a 8"
    #reference_path = generate_straight_and_turn_path(straight_len=3.0, turn_radius=1.5, num_points=200)
    #trajectory_name = "Traiettoria dritta con curva"
    #reference_path = generate_random_spline_path(num_waypoints=8, num_points=300)
    #trajectory_name = "Traiettoria casuale"

    print(f"Numero waypoints: {len(reference_path)}")


    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    controller = PurePursuit(wheelbase=0.062)

    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    trajectory_x = [state.X]
    trajectory_y = [state.Y]
    time_history = [0.0]
    vx_history = [state.vx]
    vy_history = [state.vy]
    omega_history = [state.omega]
    d_history = [0.0]

    print(f"\nInizio simulazione: {num_steps} step, dt={dt}s \n" )

    l_val = controller.L_base
    vx_des = 1.5 # velocità desiderata (valore che decido io)

    kp = 10  # guadagno puramente proporzionaleora
    omega_cross = 1  # frequenza cross che mi definisce il tempo Ti ovvero, il tempo in cui voglio l'errore venga annullato
    Ti = 2 / omega_cross  # calcolo il Ti come 2 / omega_cross
    ki = kp / Ti  # calcolo il fattore di integrazione come rapporto tra il guadagno puramente proporzionale e il tempo Ti
    error_inte = 0.0

    for step in range(num_steps):


        current_time = step * dt

        delta = controller.pure_pursuit(state, reference_path) #il controllore PurePursuit mi restituisce l'angolo delta, ovvero di quanto girare le ruote




        #d_cmd = error_vx * kp  # come si calcolava d con il solo controllore puramente proporzionale

        error_vx = vx_des - state.vx

        error_inte += error_vx * dt

        d_cmd = error_inte * ki + kp * error_vx #il nuovo duty cycle è uguale alla somma dei due contributi quello puramente proporzionale moltiplicato per l'errore su vx e quello integrativo moltiplicato per error_inte


        d_cmd = np.clip(d_cmd, 0, 1.0)


        control_input = VehicleInput(d=d_cmd , delta=delta)
        control_input.saturate()



        state = integrator.Eulero(state, control_input)

        trajectory_x.append(state.X)
        trajectory_y.append(state.Y)
        time_history.append(current_time + dt)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        omega_history.append(state.omega)
        d_history.append(d_cmd)

        if step % int(2 / dt) == 0:
            print(f"t={current_time:.1f}s: pos=({state.X:.3f}, {state.Y:.3f}), " #se rientro nel tempo, stampo il current_time , la posizione X e Y e vx e vy
                  f"vx={state.vx:.3f}, vy={state.vy:.4f}")

    print("\nSimulazione completata!")




    errors = []  #dichiaro array errori
    for x, y in zip(trajectory_x, trajectory_y):
        min_dist = min(np.sqrt((x - rx) ** 2 + (y - ry) ** 2) for rx, ry in reference_path) #calcolo la distanza da ogni punto della traiettoria e prendo la minima distanza dal punto più vicino del path
        errors.append(min_dist)

    print("\n=========== STATISTICHE FINE SIMULAZIONE ===========")
    print(f"Errore medio: {np.mean(errors):.4f} m")  #calcolo e restituisco la media degli errori con la libreria numpy
    print(f"Errore massimo: {np.max(errors):.4f} m") #restituisco il massimo
    print(f"Errore quadratico medio: {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m")
    print("====================================================")

    # ==============================================================
    #                PLOT
    # ==============================================================

    base_dir = "Grafici"
    tipo_controllo = "azione_PI"
    ora_attuale = datetime.datetime.now().strftime("%H%M%S")
    dettagli_run = f"Vel{vx_des}_Kp{kp}_Ki{ki:.2f}_Ti{Ti:.2f}_T{T_sim}_L{l_val}{ora_attuale}"


    run_dir = os.path.join(base_dir, trajectory_name, tipo_controllo, dettagli_run)


    os.makedirs(run_dir, exist_ok=True)


    path_file_testo = os.path.join(run_dir, "parametri_simulazione.txt")
    with open(path_file_testo, "w") as f:
        f.write("DATI DELL'ESPERIMENTO\n")
        f.write("=====================\n")
        f.write(f"Traiettoria: {trajectory_name}\n")
        f.write(f"Tempo totale (T_sim): {T_sim} s\n")
        f.write(f"Target Velocità (vx_des): {vx_des} m/s\n")
        f.write(f"Guadagno Proporzionale (kp): {kp}\n")
        f.write(f"Passo integrazione (dt): {dt} s\n")

    """run_name = f"T{T_sim}_dt{dt}_d{d_constant}"
    base_dir = "Grafici"
    traj_dir = os.path.join(base_dir, trajectory_name)  #sottocartella per traiettoria
    run_dir = os.path.join(traj_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)"""



    #si separano x e y in due liste
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
    plt.savefig(os.path.join(run_dir, "Traiettoria_riferimento.png"), dpi = 300)
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
    plt.savefig(os.path.join(run_dir, "Traiettoria_Seguita.png"), dpi = 300)
    plt.show()

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex = True) #5 figure all'interno del file una per vx(t), una per vy(t), una per w(t) e una per l'errore

    axes[0].plot(time_history, vx_history, label = 'vx reale')
    axes[0].axhline(y = vx_des, color = 'brown', linestyle = '--', linewidth = 1.5, label = 'vx des')
    axes[0].set_ylabel('vx [m/s]')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_history, vy_history)
    axes[1].set_ylabel('vy [m/s]')
    axes[1].grid(True)

    axes[2].plot(time_history, omega_history)
    axes[2].set_ylabel('ω [rad/s]')
    axes[2].grid(True)

    axes[3].plot(time_history, errors)
    axes[3].set_ylabel('Errore [m]')
    axes[3].grid(True)

    axes[4].plot(time_history, d_history, color='orange', linewidth=1.5)
    axes[4].set_ylabel('Duty Cycle [0-1]')
    axes[4].set_xlabel('Tempo [s]')
    axes[4].set_ylim(-0.1, 1.1)  # Per vedere bene le saturazioni
    axes[4].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "Stati_e_Errore.png"), dpi = 300)
    plt.show()

    video_path = os.path.join(run_dir, "simulazione_traiettoria.mp4")
    simulazione_traiettoria(ref_x, ref_y, trajectory_x, trajectory_y, dt, video_path)



if __name__ == "__main__":
   run_simulation()
   #run_vx_test()
   #run_open_loop_step_test()

