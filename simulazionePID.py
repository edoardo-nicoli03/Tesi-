"""
Simulazione con controllo in cascata:
Pure Pursuit → omega_des → PID laterale → delta

Questo file implementa il nuovo approccio dove:
1. Pure Pursuit calcola la velocità angolare desiderata (omega_des)
2. Un PID laterale controlla lo sterzo per raggiungere omega_des
3. Un PI longitudinale controlla la velocità

TUNING PROGRESSIVO:
- Parti con solo P sul PID laterale
- Aggiungi I se c'è errore stazionario
- Aggiungi D se oscilla troppo
"""

import matplotlib

matplotlib.use('TkAgg')
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.interpolate import splev, splprep
import datetime

# Import moduli veicolo
from Veicolo.Vehicle_model import (VehicleIntegrator, VehicleInput,
                                   VehicleState, DynamicBicycleModel)
from Veicolo.Vehicle_controller import PurePursuit
from Veicolo.PID_controller import YawRatePIDController, VelocityPIDController


# ==============================================================
#            FUNZIONI PER GENERARE TRAIETTORIE
# ==============================================================

def generate_circular_path(radius: float, center: Tuple[float, float],
                           num_points: int = 150) -> List[Tuple[float, float]]:
    """Genera un percorso circolare"""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points)
    path = [(cx + radius * np.cos(angle), cy + radius * np.sin(angle))
            for angle in angles]
    return path


def generate_figure_eight_path(width: float, num_points: int = 200) -> List[Tuple[float, float]]:
    """Genera un percorso a forma di 8"""
    t = np.linspace(0, 2 * np.pi, num_points)
    x = width * np.sin(t)
    y = width * np.sin(t) * np.cos(t)
    path = [(x[i], y[i]) for i in range(num_points)]
    return path


def generate_circuite_path(waypoints: List[Tuple[float, float]],
                           num_points: int = 300,
                           smoothness: int = 3) -> List[Tuple[float, float]]:
    """Genera un circuito interpolando waypoints con spline cubica"""
    waypoints_array = np.array(waypoints)
    x = waypoints_array[:, 0]
    y = waypoints_array[:, 1]

    tck, u = splprep([x, y], s=0, k=smoothness, per=1)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
    return path


# ==============================================================
#                    SIMULAZIONE PRINCIPALE
# ==============================================================

def run_simulation_cascata():
    """
    Esegue la simulazione con controllo in cascata.

    Architettura:
    Pure Pursuit → omega_des → PID(omega) → delta
                                            ↓
    PI(vx) → d →  Veicolo  → stato → feedback
    """

    # ===== PARAMETRI SIMULAZIONE =====
    dt = 0.001  # Passo di integrazione [s]
    T_sim = 40.0  # Durata simulazione [s]
    num_steps = int(T_sim / dt)

    print("=" * 70)
    print("SIMULAZIONE CON CONTROLLO IN CASCATA")
    print("Pure Pursuit -> PID laterale (ω) -> δ")
    print("=" * 70)

    # ===== TRAIETTORIA DI RIFERIMENTO =====
    track_waypoints = [
        (0.0, 0.0), (0.5, 1.2), (1.2, 2.2),
        (2.5, 3.0), (4.5, 3.2), (6.5, 3.2), (8.2, 3.0),
        (9.0, 2.2), (9.2, 1.2),
        (8.5, 0.3), (6.8, -0.3), (4.7, -0.5),
        (3.5, -0.8), (2.3, -1.4),
        (0.5, -1.6), (-1.2, -1.4), (-2.0, -0.8),
        (-2.2, 0.0), (-1.5, 0.8), (-0.5, 1.0),
        (1.1, -0.6), (2.1, 0.0),
    ]

    reference_path = generate_circuite_path(track_waypoints, num_points=300)
    trajectory_name = "Pista_Racing"

    #Altre traiettorie disponibili:
    #reference_path = generate_circular_path(radius=2.0, center=(2.0, 0.0), num_points=150)
    #trajectory_name = "Circolare"

    #reference_path = generate_figure_eight_path(width=4.0, num_points=200)
    #trajectory_name = "Otto"

    print(f"\nTraiettoria: {trajectory_name}")
    print(f"Numero waypoints: {len(reference_path)}")

    # ===== MODELLO E INTEGRATORE =====
    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    # ===== CONTROLLORE PURE PURSUIT =====
    pure_pursuit = PurePursuit(
        wheelbase=0.062,
        lookahead_base=0.15,
        lookahead_gain=0.12,
        max_yaw_rate=5.0  # Limite velocità angolare [rad/s]
    )

    # ===== CONTROLLORE LONGITUDINALE (velocità) =====
    vx_des = 1.5  # Velocità desiderata [m/s]

    pid_longitudinal = VelocityPIDController(
        kp=10.0,
        ki=5.0,
        dt=dt,
        max_duty=1.0
    )

    # ===== CONTROLLORE LATERALE (velocità angolare) =====
    # CONFIGURAZIONE TUNING PROGRESSIVO
    #
    # TEST 1: Solo P
    kp_omega = 0.3
    ki_omega = 0.0
    kd_omega = 0.0
    use_integral = False
    use_derivative = False

    # TEST 2: PI (decommenta per provare)
    # kp_omega = 0.3
    # ki_omega = 0.01
    # kd_omega = 0.0
    # use_integral = True
    # use_derivative = False

    # TEST 3: PID (decommenta per provare)
    # kp_omega = 0.3
    # ki_omega = 0.01
    # kd_omega = 0.05
    # use_integral = True
    # use_derivative = True

    pid_lateral = YawRatePIDController(
        kp=kp_omega,
        ki=ki_omega,
        kd=kd_omega,
        dt=dt,
        max_delta=0.35,
        use_integral=use_integral,
        use_derivative=use_derivative
    )

    control_type = pid_lateral.get_control_type()

    print(f"\n{'=' * 70}")
    print(f"CONFIGURAZIONE CONTROLLORI")
    print(f"{'=' * 70}")
    print(f"Longitudinale (vx -> d): PI")
    print(f"  - vx_des: {vx_des} m/s")
    print(f"  - Kp: {pid_longitudinal.kp}")
    print(f"  - Ki: {pid_longitudinal.ki}")
    print(f"\nLaterale (ω -> δ): {control_type}")
    print(f"  - Kp: {pid_lateral.kp}")
    if use_integral:
        print(f"  - Ki: {pid_lateral.ki}")
    if use_derivative:
        print(f"  - Kd: {pid_lateral.kd}")
    print(f"{'=' * 70}\n")

    # ===== STATO INIZIALE =====
    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    # ===== ARRAY PER LOGGING =====
    trajectory_x = [state.X]
    trajectory_y = [state.Y]
    time_history = [0.0]
    vx_history = [state.vx]
    vy_history = [state.vy]
    omega_history = [state.omega]
    omega_des_history = [0.0]  # Velocità angolare desiderata
    delta_history = [0.0]
    d_history = [0.0]

    # ===== LOOP DI SIMULAZIONE =====
    print(f"Inizio simulazione: {num_steps} step, dt={dt}s")
    print("Questo potrebbe richiedere qualche secondo...\n")

    for step in range(num_steps):
        current_time = step * dt

        # ========================================
        # 1. PURE PURSUIT → omega_des
        # ========================================
        omega_des = pure_pursuit.yaw_rate_des(state, reference_path)

        # ========================================
        # 2. PID LATERALE: omega → delta
        # ========================================
        delta = pid_lateral.compute(omega_des, state.omega)

        # ========================================
        # 3. PID LONGITUDINALE: vx → d
        # ========================================
        d_cmd = pid_longitudinal.compute(vx_des, state.vx)

        # ========================================
        # 4. APPLICA CONTROLLO AL VEICOLO
        # ========================================
        control_input = VehicleInput(d=d_cmd, delta=delta)
        control_input.saturate()

        # ========================================
        # 5. INTEGRA DINAMICA
        # ========================================
        state = integrator.Eulero(state, control_input)

        # ========================================
        # 6. LOGGING
        # ========================================
        trajectory_x.append(state.X)
        trajectory_y.append(state.Y)
        time_history.append(current_time + dt)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        omega_history.append(state.omega)
        omega_des_history.append(omega_des)
        delta_history.append(delta)
        d_history.append(d_cmd)

        # Stampa progresso ogni 2 secondi
        if step % int(2 / dt) == 0:
            print(f"t={current_time:.1f}s: pos=({state.X:.3f}, {state.Y:.3f}), "
                  f"vx={state.vx:.3f}, ω={state.omega:.3f} (ω_des={omega_des:.3f})")

    print("\n✓ Simulazione completata!")

    # ===== CALCOLA STATISTICHE ERRORE =====
    errors = []
    for x, y in zip(trajectory_x, trajectory_y):
        min_dist = min(np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
                       for rx, ry in reference_path)
        errors.append(min_dist)

    print(f"\n{'=' * 70}")
    print(f"STATISTICHE FINALI")
    print(f"{'=' * 70}")
    print(f"Errore medio:     {np.mean(errors):.4f} m")
    print(f"Errore massimo:   {np.max(errors):.4f} m")
    print(f"Errore RMS:       {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m")
    print(f"{'=' * 70}\n")

    # ==============================================================
    #                    SALVATAGGIO RISULTATI
    # ==============================================================

    base_dir = "Grafici"
    ora_attuale = datetime.datetime.now().strftime("%H%M%S")

    # Nome cartella basato su tipo controllore e parametri
    if control_type == "P":
        dettagli_run = f"Cascata_{control_type}_Kp{kp_omega:.2f}_vx{vx_des}_{ora_attuale}"
    elif control_type == "PI":
        dettagli_run = f"Cascata_{control_type}_Kp{kp_omega:.2f}_Ki{ki_omega:.3f}_vx{vx_des}_{ora_attuale}"
    elif control_type == "PD":
        dettagli_run = f"Cascata_{control_type}_Kp{kp_omega:.2f}_Kd{kd_omega:.3f}_vx{vx_des}_{ora_attuale}"
    else:  # PID
        dettagli_run = f"Cascata_{control_type}_Kp{kp_omega:.2f}_Ki{ki_omega:.3f}_Kd{kd_omega:.3f}_vx{vx_des}_{ora_attuale}"

    run_dir = os.path.join(base_dir, trajectory_name, "controllo_cascata", dettagli_run)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Salvataggio risultati in: {run_dir}\n")

    # Salva parametri simulazione in file di testo
    with open(os.path.join(run_dir, "parametri_simulazione.txt"), "w", encoding='utf-8') as f:
        f.write("CONTROLLO IN CASCATA\n")
        f.write("=" * 70 + "\n")
        f.write("Architettura:\n")
        f.write("  Pure Pursuit -> omega_des -> PID laterale -> delta\n")
        f.write("  PI longitudinale -> duty cycle\n\n")

        f.write(f"Traiettoria: {trajectory_name}\n")
        f.write(f"Tempo simulazione: {T_sim} s\n")
        f.write(f"dt: {dt} s\n")
        f.write(f"Numero step: {num_steps}\n\n")

        f.write("CONTROLLORE LONGITUDINALE (vx -> d):\n")
        f.write(f"  Tipo: PI\n")
        f.write(f"  vx_des: {vx_des} m/s\n")
        f.write(f"  Kp: {pid_longitudinal.kp}\n")
        f.write(f"  Ki: {pid_longitudinal.ki}\n\n")

        f.write("CONTROLLORE LATERALE (ω -> δ):\n")
        f.write(f"  Tipo: {control_type}\n")
        f.write(f"  Kp: {pid_lateral.kp}\n")
        if use_integral:
            f.write(f"  Ki: {pid_lateral.ki}\n")
        if use_derivative:
            f.write(f"  Kd: {pid_lateral.kd}\n")

        f.write("\nSTATISTICHE:\n")
        f.write(f"  Errore medio: {np.mean(errors):.4f} m\n")
        f.write(f"  Errore max: {np.max(errors):.4f} m\n")
        f.write(f"  Errore RMS: {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m\n")

    # ==============================================================
    #                    GENERAZIONE GRAFICI
    # ==============================================================

    print("Generazione grafici...")

    ref_x = [p[0] for p in reference_path]
    ref_y = [p[1] for p in reference_path]

    # ===== GRAFICO 1: TRAIETTORIA XY =====
    plt.figure(figsize=(10, 8))
    plt.plot(ref_x, ref_y, 'r--', linewidth=2, label='Traiettoria riferimento')
    plt.plot(trajectory_x, trajectory_y, 'b-', linewidth=1.8, label='Traiettoria seguita')
    plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Start', zorder=5)
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'rs', markersize=10, label='End', zorder=5)
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Y [m]', fontsize=12)
    plt.title(f'Path Tracking - Controllo Cascata ({control_type})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "01_Traiettoria.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ===== GRAFICO 2: STATI E COMANDI =====
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)

    # Subplot 1: Velocità longitudinale
    axes[0].plot(time_history, vx_history, 'b-', label='vx reale', linewidth=2)
    axes[0].axhline(y=vx_des, color='r', linestyle='--', linewidth=1.5, label='vx desiderato')
    axes[0].set_ylabel('vx [m/s]', fontsize=11)
    axes[0].set_title('Velocità longitudinale', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9)

    # Subplot 2: Velocità angolare (IMPORTANTE!)
    axes[1].plot(time_history, omega_history, 'b-', label='ω reale', linewidth=2)
    axes[1].plot(time_history, omega_des_history, 'r--', label='ω desiderato', linewidth=1.5)
    axes[1].set_ylabel('ω [rad/s]', fontsize=11)
    axes[1].set_title(f'Velocità angolare - Controllo {control_type}', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9)

    # Subplot 3: Angolo di sterzo
    axes[2].plot(time_history, delta_history, 'purple', linewidth=1.5)
    axes[2].axhline(y=0.35, color='gray', linestyle=':', linewidth=1, label='Limiti fisici')
    axes[2].axhline(y=-0.35, color='gray', linestyle=':', linewidth=1)
    axes[2].set_ylabel('δ [rad]', fontsize=11)
    axes[2].set_title('Angolo di sterzo', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=9)

    # Subplot 4: Velocità laterale
    axes[3].plot(time_history, vy_history, 'g-', linewidth=1.5)
    axes[3].set_ylabel('vy [m/s]', fontsize=11)
    axes[3].set_title('Velocità laterale', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    # Subplot 5: Errore di tracking
    axes[4].plot(time_history, errors, 'orange', linewidth=1.5)
    axes[4].set_ylabel('Errore [m]', fontsize=11)
    axes[4].set_title('Errore laterale rispetto alla traiettoria', fontsize=12, fontweight='bold')
    axes[4].grid(True, alpha=0.3)

    # Subplot 6: Duty cycle
    axes[5].plot(time_history, d_history, 'brown', linewidth=1.5)
    axes[5].set_ylabel('Duty Cycle', fontsize=11)
    axes[5].set_xlabel('Tempo [s]', fontsize=12)
    axes[5].set_title('Comando motore', fontsize=12, fontweight='bold')
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "02_Stati_e_Comandi.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ===== GRAFICO 3: ANALISI ERRORE OMEGA =====
    omega_error = [omega_des_history[i] - omega_history[i]
                   for i in range(len(omega_history))]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time_history, omega_des_history, 'r--', label='ω desiderato', linewidth=1.5)
    axes[0].plot(time_history, omega_history, 'b-', label='ω reale', linewidth=2)
    axes[0].set_ylabel('ω [rad/s]', fontsize=11)
    axes[0].set_title('Inseguimento velocità angolare', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)

    axes[1].plot(time_history, omega_error, 'red', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Errore ω [rad/s]', fontsize=11)
    axes[1].set_xlabel('Tempo [s]', fontsize=12)
    axes[1].set_title(f'Errore su ω (Controllo {control_type})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "03_Analisi_Omega.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Grafici salvati in: {run_dir}\n")

    # Mostra i grafici
    print("Apertura grafici...")

    # Riapri e mostra i grafici principali
    img1 = plt.imread(os.path.join(run_dir, "01_Traiettoria.png"))
    img2 = plt.imread(os.path.join(run_dir, "02_Stati_e_Comandi.png"))
    img3 = plt.imread(os.path.join(run_dir, "03_Analisi_Omega.png"))

    plt.figure(figsize=(12, 10))
    plt.imshow(img1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 15))
    plt.imshow(img2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.imshow(img3)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("SIMULAZIONE TERMINATA")
    print("=" * 70)
    print(f"\nRisultati salvati in: {run_dir}")
    print("\nPer provare configurazioni diverse, modifica i parametri")
    print("del PID laterale nella sezione 'CONFIGURAZIONE TUNING PROGRESSIVO'")
    print("=" * 70)


# ==============================================================
#                    ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    run_simulation_cascata()