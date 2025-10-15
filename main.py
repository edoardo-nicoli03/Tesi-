import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.interpolate import CubicSpline
from Vehicle_model import VehicleIntegrator, VehicleInput, VehicleState, DynamicBicycleModel
from Vehicle_controller import PurePursuit


# SCRIPT DI SIMULAZIONE


""" GENERA UNA TRAIETTORIA A FORMA DI CERCHIO"""

"""def generate_circular_path(radius: float, center: Tuple[float, float],
                           num_points: int = 150) -> List[Tuple[float, float]]:
    
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points)

    path = []
    for angle in angles:
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        path.append((x, y))


    return path """


"""  GENERA UNA TRAIETTORIA A FORMA DI 8
def generate_figure_eight_path(width: float, num_points: int = 200) -> List[Tuple[float, float]]:
  
    path = []
    t = np.linspace(0, 2 * np.pi, num_points)

    # Formula della Lemniscata di Gerono
    x = width * np.sin(t)
    y = width * np.sin(t) * np.cos(t)

    for i in range(num_points):
        path.append((x[i], y[i]))

    return path """

"""Genera un rettilineo seguito da una curva a 90 gradi.
def generate_straight_and_turn_path(straight_len: float, turn_radius: float, num_points: int = 150) -> List[
    Tuple[float, float]]:
   
    path = []

    # Parte 1: Linea retta
    num_straight = int(num_points * 0.6)
    for x in np.linspace(0, straight_len, num_straight):
        path.append((x, 0.0))

    # Parte 2: Curva a 90 gradi
    num_turn = num_points - num_straight
    center_x = straight_len
    center_y = turn_radius

    for angle in np.linspace(-np.pi / 2, 0, num_turn):
        x = center_x + turn_radius * np.cos(angle)
        y = center_y + turn_radius * np.sin(angle)
        path.append((x, y))

    return path """
def generate_random_spline_path(num_waypoints: int = 7,
                                x_range: Tuple[float, float] = (0, 10),
                                y_range: Tuple[float, float] = (-3, 3),
                                num_points: int = 300) -> List[Tuple[float, float]]:


    waypoints_x = np.linspace(x_range[0], x_range[1], num_waypoints)
    waypoints_y = np.random.uniform(y_range[0], y_range[1], num_waypoints)

  #punti di partenza della traiettoria
    waypoints_x[0] = 0
    waypoints_y[0] = 0

    # Creo la funzione di interpolazione
    spline = CubicSpline(waypoints_x, waypoints_y)

    #  Campiono la curva per creare la traiettoria finale
    x_new = np.linspace(waypoints_x[0], waypoints_x[-1], num_points)
    y_new = spline(x_new) # Calcola le y corrispondenti sulla curva


    path = []
    for i in range(len(x_new)):
        path.append((x_new[i], y_new[i]))

    return path


def run_simulation():

    # parametri di simulazione
    dt = 0.01  # tempo di campionamento [s]
    T_sim = 20.0  # durata simulazione [s]
    num_steps = int(T_sim / dt)

    # Duty cycle costante
    d_constant = 0.25

  # GENERA TRAIETTORIA

    radius = 2.0  # [m]
    center = (2.0, 0.0)  # centro in (2, 0) -> parte da (0, 0)
    #reference_path per le traiettorie di Cerchio, 8, dritto con curva
   # reference_path = generate_circular_path(radius, center, num_points=150) #Per cerchio
   # reference_path = generate_figure_eight_path(width=4.0, num_points=200)  #Per otto
    #reference_path = generate_straight_and_turn_path(straight_len=3.0, turn_radius=1.5, num_points=200)
    reference_path = generate_random_spline_path(num_waypoints=8, num_points=300)


    print(f"Numero di waypoints: {len(reference_path)}")


    # INIZIALIZZA MODELLO E CONTROLLORE

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    controller = PurePursuit(wheelbase=0.062)
    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)


    # SIMULAZIONE

    print(f"\nInizio simulazione: {num_steps} step, dt={dt}s")

    # Storage per i risultati
    trajectory_x = [state.X]
    trajectory_y = [state.Y]
    time_history = [0.0]
    vx_history = [state.vx]
    vy_history = [state.vy]
    omega_history = [state.omega]

    for step in range(num_steps):
        current_time = step * dt


        delta = controller.pure_pursuit(state, reference_path) #delta preso dal Pure_Pursuit

        # Crea input di controllo (duty cycle costante)
        control_input = VehicleInput(d=d_constant, delta=delta)
        control_input.saturate() #verifico che rispettino i valori di saturazione

        # Integrazione numerica con Eulero
        state = integrator.Eulero(state, control_input)

        trajectory_x.append(state.X)
        trajectory_y.append(state.Y)
        time_history.append(current_time + dt)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        omega_history.append(state.omega)

        # Stampo progresso
        if step % int(2.0 / dt) == 0:
            print(f"t={current_time:.1f}s: pos=({state.X:.3f}, {state.Y:.3f}), "
                  f"vx={state.vx:.3f}, vy={state.vy:.4f}")

    print("\nSimulazione completata!")


    # VISUALIZZAZIONE RISULTATI

    """
    # Estrai coordinate della traiettoria di riferimento
    ref_x = [p[0] for p in reference_path]
    ref_y = [p[1] for p in reference_path]

    # --- GRAFICO 1: Traiettoria XY ---
    plt.figure(figsize=(10, 8))

    plt.plot(ref_x, ref_y, 'r--', linewidth=2, label='Traiettoria di riferimento')
    plt.plot(trajectory_x, trajectory_y, 'b-', linewidth=1.5, label='Traiettoria effettiva')
    plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Partenza')
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'rs', markersize=10, label='Arrivo')

    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Y [m]', fontsize=12)
    plt.title('Traiettoria del Veicolo - Pure Pursuit su Percorso Circolare', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    # --- GRAFICO 2: Stati nel Tempo ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Velocità longitudinale
    axes[0].plot(time_history, vx_history, 'b-', linewidth=1.5)
    axes[0].set_ylabel('vx [m/s]', fontsize=11)
    axes[0].set_title('Evoluzione Temporale degli Stati del Veicolo', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Velocità longitudinale'], fontsize=10)

    # Velocità laterale (dovrebbe rimanere vicino a zero - vincolo no-slip)
    axes[1].plot(time_history, vy_history, 'r-', linewidth=1.5)
    axes[1].set_ylabel('vy [m/s]', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(['Velocità laterale (no-slip)'], fontsize=10)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)

    # Velocità angolare
    axes[2].plot(time_history, omega_history, 'g-', linewidth=1.5)
    axes[2].set_xlabel('Tempo [s]', fontsize=11)
    axes[2].set_ylabel('ω [rad/s]', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(['Velocità angolare'], fontsize=10)

    plt.tight_layout()
    """



    # STATISTICHE FINALI

    print("\n" + "=" * 60)
    print("STATISTICHE SIMULAZIONE")
    print("=" * 60)
    print(f"Velocità longitudinale media: {np.mean(vx_history):.3f} m/s")
    print(f"Velocità longitudinale finale: {vx_history[-1]:.3f} m/s")
    print(f"Velocità laterale max (abs): {np.max(np.abs(vy_history)):.5f} m/s")
    print(f"Velocità laterale RMS: {np.sqrt(np.mean(np.array(vy_history) ** 2)):.5f} m/s")
    print(f"Velocità angolare max (abs): {np.max(np.abs(omega_history)):.3f} rad/s")

    # Calcolo errore rispetto alla traiettoria
    errors = []
    for x, y in zip(trajectory_x, trajectory_y):
        min_dist = float('inf')
        for ref_x_pt, ref_y_pt in reference_path:
            dist = np.sqrt((x - ref_x_pt) ** 2 + (y - ref_y_pt) ** 2)
            if dist < min_dist:
                min_dist = dist
        errors.append(min_dist)

    print(f"\nErrore di tracking medio: {np.mean(errors):.4f} m")
    print(f"Errore di tracking max: {np.max(errors):.4f} m")
    print(f"Errore di tracking RMS: {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    run_simulation()