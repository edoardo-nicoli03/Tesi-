"""
FASE 1: Pure Pursuit con duty cycle costante

Configurazione:
- Pure Pursuit calcola delta direttamente
- Duty cycle fisso d = 0.3
- Velocità NON controllata (variabile)

Traiettorie disponibili:
- Pista Racing (default)
- Traiettoria Circolare
- Traiettoria a 8
"""

import matplotlib

matplotlib.use('TkAgg')
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.interpolate import splev, splprep
import datetime

from Veicolo.Vehicle_model import (VehicleIntegrator, VehicleInput,
                                   VehicleState, DynamicBicycleModel)
from Veicolo.Vehicle_controller import PurePursuit


# ==============================================================
#            FUNZIONI PER GENERARE TRAIETTORIE
# ==============================================================

def generate_circular_path(radius: float, center: Tuple[float, float],
                           num_points: int = 150) -> List[Tuple[float, float]]:
    """Genera percorso circolare"""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points)

    path = []
    for angle in angles:
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        path.append((x, y))

    return path


def generate_figure_eight_path(width: float, num_points: int = 200) -> List[Tuple[float, float]]:
    """Genera percorso a forma di 8"""
    path = []
    t = np.linspace(0, 2 * np.pi, num_points)

    x = width * np.sin(t)
    y = width * np.sin(t) * np.cos(t)

    for i in range(num_points):
        path.append((x[i], y[i]))

    return path


def generate_circuite_path(waypoints: List[Tuple[float, float]],
                           num_points: int = 300,
                           smoothness: int = 3) -> List[Tuple[float, float]]:
    """Genera circuito interpolando waypoints con spline cubica"""
    waypoints_array = np.array(waypoints)
    x = waypoints_array[:, 0]
    y = waypoints_array[:, 1]

    tck, u = splprep([x, y], s=0, k=smoothness, per=1)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
    return path


# ==============================================================
#                    SIMULAZIONE FASE 1
# ==============================================================

def run_fase1_simulation():
    """
    Esegue simulazione Fase 1: Pure Pursuit + d costante
    """

    # ===== PARAMETRI SIMULAZIONE =====
    dt = 0.001  # Passo di integrazione [s]
    T_sim = 40.0  # Durata simulazione [s]
    num_steps = int(T_sim / dt)

    d_constant = 0.3  # ← DUTY CYCLE COSTANTE

    print("=" * 70)
    print("FASE 1: PURE PURSUIT + DUTY CYCLE COSTANTE")
    print("=" * 70)

    # ===== TRAIETTORIA DI RIFERIMENTO =====
    # Definisci waypoints pista racing
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

    # SCEGLI LA TRAIETTORIA (commenta/decommenta)
    reference_path = generate_circuite_path(track_waypoints, num_points=300)
    trajectory_name = "Pista_Racing"

    # reference_path = generate_circular_path(radius=2.0, center=(2.0, 0.0), num_points=150)
    # trajectory_name = "Traiettoria_Circolare"

    # reference_path = generate_figure_eight_path(width=4.0, num_points=200)
    # trajectory_name = "Traiettoria_a_8"

    print(f"\nTraiettoria: {trajectory_name}")
    print(f"Waypoints: {len(reference_path)}")

    # ===== MODELLO E CONTROLLORE =====
    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    controller = PurePursuit(
        wheelbase=0.062,
        lookahead_base=0.15,  # ← VALORI DALLA TESI
        lookahead_gain=0.12
    )

    # ===== STATO INIZIALE =====
    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    # ===== LOGGING =====
    trajectory_x = [state.X]
    trajectory_y = [state.Y]
    time_history = [0.0]
    vx_history = [state.vx]
    vy_history = [state.vy]
    omega_history = [state.omega]
    delta_history = [0.0]
    d_history = [d_constant]

    print(f"\n{'=' * 70}")
    print("CONFIGURAZIONE")
    print(f"{'=' * 70}")
    print(f"Controllo laterale: Pure Pursuit → delta (DIRETTO)")
    print(f"Controllo longitudinale: NESSUNO")
    print(f"Duty cycle: {d_constant} (COSTANTE)")
    print(f"Velocità: NON controllata (varierà)")
    print(f"Lookahead base: 0.15 m")
    print(f"Lookahead gain: 0.12")
    print(f"{'=' * 70}\n")

    print(f"Inizio simulazione: {num_steps} step, dt={dt}s\n")

    # ===== LOOP SIMULAZIONE =====
    for step in range(num_steps):
        current_time = step * dt

        # ========================================
        # 1. PURE PURSUIT → delta (DIRETTO)
        # ========================================
        delta = controller.pure_pursuit(state, reference_path)

        # ========================================
        # 2. INPUT: d costante, delta da PP
        # ========================================
        control_input = VehicleInput(d=d_constant, delta=delta)
        control_input.saturate()

        # ========================================
        # 3. INTEGRA DINAMICA
        # ========================================
        state = integrator.Eulero(state, control_input)

        # ========================================
        # 4. LOGGING
        # ========================================
        trajectory_x.append(state.X)
        trajectory_y.append(state.Y)
        time_history.append(current_time + dt)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        omega_history.append(state.omega)
        delta_history.append(delta)
        d_history.append(d_constant)

        # Stampa progresso ogni 2 secondi
        if step % int(2 / dt) == 0:
            print(f"t={current_time:.1f}s: pos=({state.X:.3f}, {state.Y:.3f}), "
                  f"vx={state.vx:.3f}")

    print("\n✓ Simulazione completata!")

    # ===== CALCOLA STATISTICHE ERRORE =====
    errors = []
    for x, y in zip(trajectory_x, trajectory_y):
        min_dist = min(np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
                       for rx, ry in reference_path)
        errors.append(min_dist)

    print(f"\n{'=' * 70}")
    print("STATISTICHE FINALI")
    print(f"{'=' * 70}")
    print(f"Errore medio:     {np.mean(errors):.4f} m")
    print(f"Errore massimo:   {np.max(errors):.4f} m")
    print(f"Errore RMS:       {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m")
    print(f"\nVelocità longitudinale:")
    print(f"  vx media:       {np.mean(vx_history):.3f} m/s")
    print(f"  vx min:         {np.min(vx_history):.3f} m/s")
    print(f"  vx max:         {np.max(vx_history):.3f} m/s")
    print(f"  vx std dev:     {np.std(vx_history):.3f} m/s")
    print(f"{'=' * 70}\n")

    # ==============================================================
    #                    SALVATAGGIO RISULTATI
    # ==============================================================

    base_dir = "Grafici"

    ora_attuale = datetime.datetime.now().strftime("%H%M%S")
    L_base = controller.L_base
    L_gain = controller.L_gain
    dettagli_run = f"d{d_constant}_{ora_attuale}_Lb{L_base}_Lg{L_gain}"

    run_dir = os.path.join(base_dir, trajectory_name, "Fase1_PP_solo", dettagli_run)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Salvataggio risultati in: {run_dir}\n")

    # Salva parametri simulazione
    with open(os.path.join(run_dir, "parametri_simulazione.txt"), "w", encoding='utf-8') as f:
        f.write("FASE 1: PURE PURSUIT + DUTY CYCLE COSTANTE\n")
        f.write("=" * 70 + "\n")
        f.write(f"Traiettoria: {trajectory_name}\n")
        f.write(f"Tempo simulazione: {T_sim} s\n")
        f.write(f"dt: {dt} s\n")
        f.write(f"Numero step: {num_steps}\n\n")

        f.write("CONFIGURAZIONE CONTROLLO:\n")
        f.write("  Controllo laterale: Pure Pursuit → delta (DIRETTO)\n")
        f.write("  Controllo longitudinale: NESSUNO\n")
        f.write(f"  Duty cycle: {d_constant} (COSTANTE)\n")
        f.write(f"  Lookahead base: 0.15 m\n")
        f.write(f"  Lookahead gain: 0.12\n\n")

        f.write("STATISTICHE:\n")
        f.write(f"  Errore medio: {np.mean(errors):.4f} m\n")
        f.write(f"  Errore max: {np.max(errors):.4f} m\n")
        f.write(f"  Errore RMS: {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m\n\n")
        f.write(f"  vx media: {np.mean(vx_history):.3f} m/s\n")
        f.write(f"  vx range: [{np.min(vx_history):.3f}, {np.max(vx_history):.3f}] m/s\n")
        f.write(f"  vx std dev: {np.std(vx_history):.3f} m/s\n")

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
    plt.title('Fase 1: Pure Pursuit + d costante', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "01_Traiettoria.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ===== GRAFICO 2: STATI E COMANDI =====
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    # Subplot 1: Velocità longitudinale (NON controllata)
    axes[0].plot(time_history, vx_history, 'b-', linewidth=2, label='vx (non controllata)')
    axes[0].axhline(y=np.mean(vx_history), color='gray', linestyle=':',
                    linewidth=1.5, label=f'Media: {np.mean(vx_history):.2f} m/s')
    axes[0].set_ylabel('vx [m/s]', fontsize=11)
    axes[0].set_title('Velocità longitudinale (NON CONTROLLATA)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9)

    # Subplot 2: Velocità laterale
    axes[1].plot(time_history, vy_history, 'g-', linewidth=1.5)
    axes[1].set_ylabel('vy [m/s]', fontsize=11)
    axes[1].set_title('Velocità laterale', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Velocità angolare
    axes[2].plot(time_history, omega_history, 'purple', linewidth=1.5)
    axes[2].set_ylabel('ω [rad/s]', fontsize=11)
    axes[2].set_title('Velocità angolare', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Subplot 4: Angolo di sterzo (da Pure Pursuit diretto)
    axes[3].plot(time_history, delta_history, 'orange', linewidth=1.5)
    axes[3].axhline(y=0.35, color='gray', linestyle=':', linewidth=1, label='Limiti fisici')
    axes[3].axhline(y=-0.35, color='gray', linestyle=':', linewidth=1)
    axes[3].set_ylabel('δ [rad]', fontsize=11)
    axes[3].set_title('Angolo di sterzo (da Pure Pursuit DIRETTO)', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='best', fontsize=9)

    # Subplot 5: Errore laterale
    axes[4].plot(time_history, errors, 'red', linewidth=1.5)
    axes[4].set_ylabel('Errore [m]', fontsize=11)
    axes[4].set_xlabel('Tempo [s]', fontsize=12)
    axes[4].set_title('Errore laterale rispetto alla traiettoria', fontsize=12, fontweight='bold')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "02_Stati_e_Errore.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Grafici salvati in: {run_dir}\n")

    # Mostra i grafici
    print("Apertura grafici...")

    img1 = plt.imread(os.path.join(run_dir, "01_Traiettoria.png"))
    img2 = plt.imread(os.path.join(run_dir, "02_Stati_e_Errore.png"))

    plt.figure(figsize=(12, 10))
    plt.imshow(img1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(img2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("FASE 1 COMPLETATA")
    print("=" * 70)
    print(f"\nRisultati salvati in: {run_dir}")
    print("\nPer provare altre traiettorie, modifica le righe 89-97")
    print("=" * 70)


# ==============================================================
#                    ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    run_fase1_simulation()