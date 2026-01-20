import matplotlib

matplotlib.use('TkAgg')
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Veicolo.Vehicle_model import (VehicleIntegrator, VehicleInput,
                                   VehicleState, DynamicBicycleModel, Generator_Noise)
from Veicolo.Vehicle_controller import PurePursuit
from utils.trajectory_generator import get_trajectory
from utils.plotting_utils import (setup_results_dir, plot_sim_results,
                                  genera_video_animazione, save_metadata,
                                  plot_disturbance_analysis)


# ==============================================================
#                    SIMULAZIONE FASE 1
# ==============================================================

def run_fase1_simulation():


    # ===== PARAMETRI SIMULAZIONE =====
    dt = 0.001  # Passo di integrazione [s]
    T_sim = 40.0  # Durata simulazione [s]
    num_steps = int(T_sim / dt)

    d_constant = 0.3

    print("=" * 70)
    print("FASE 1: PURE PURSUIT + DUTY CYCLE COSTANTE")
    print("=" * 70)

    reference_path, track_name = get_trajectory('racing')

    # ===== MODELLO E CONTROLLORE =====
    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    controller = PurePursuit(
        wheelbase=0.062,
        lookahead_base=0.15,
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
    vx_disturb_history = [0.0]
    omega_disturb_history = [0.0]

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

    # ===== DISTURBI =====
    USE_DISTURBANCE = True #interruttore per attivare i disturbi

    if USE_DISTURBANCE:
        disturbance = Generator_Noise(
            disturb_vx=False,
            disturb_omega=False,
            disturb_position=True, #rumore su X e Y
            disturb_heading=True, #rumore su phi (orientamento)
            magnitude=0.15,
            magnitude_position=0.03,
            magnitude_heading=0.02,  #rumore
            disturbance_type="noise"
        )

        print(f"{'=' * 70}")
        print("DISTURBI ATTIVI")
        print(f"{'=' * 70}")
        print(f"Tipo: {disturbance.disturbance_type}")
        print(f"Su vx: ±{disturbance.magnitude} m/s")
        print(f"Su omega: ±{disturbance.magnitude} rad/s")
        print(f"Su posizione (X,Y): ±{disturbance.magnitude_position} m")
        print(f"Su orientamento (φ): ±{disturbance.magnitude_heading} rad")
        print(f"{'=' * 70}\n")
    else:
        disturbance = None
        print("DISTURBI: DISATTIVATI\n")

    print(f"Inizio simulazione: {num_steps} step, dt={dt}s\n")

    for step in range(num_steps):
        current_time = step * dt


        if disturbance is not None:
            # Disturbi su velocità
            disturb_vx = disturbance.get_disturbance(current_time, 'vx')
            disturb_omega = disturbance.get_disturbance(current_time, 'omega')

            # Disturbi su POSIZIONE
            disturb_X = disturbance.get_disturbance(current_time, 'position')
            disturb_Y = disturbance.get_disturbance(current_time, 'position')
            disturb_phi = disturbance.get_disturbance(current_time, 'heading')


            vx_measured = state.vx + disturb_vx
            omega_measured = state.omega + disturb_omega
            X_measured = state.X + disturb_X
            Y_measured = state.Y + disturb_Y
            phi_measured = state.phi + disturb_phi


            vx_disturb_history.append(disturb_vx)
            omega_disturb_history.append(disturb_omega)
        else:
            vx_measured = state.vx
            omega_measured = state.omega
            X_measured = state.X
            Y_measured = state.Y
            phi_measured = state.phi
            vx_disturb_history.append(0.0)
            omega_disturb_history.append(0.0)


        state_measured = VehicleState(
            X=X_measured,
            Y=Y_measured,
            phi=phi_measured,
            vx=vx_measured,
            vy=state.vy,
            omega=omega_measured
        )


        delta = controller.pure_pursuit(state_measured, reference_path)


        control_input = VehicleInput(d=d_constant, delta=delta)
        control_input.saturate()


        state = integrator.Eulero(state, control_input)


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

    # ===== SALVATAGGIO =====
    param_str = f"d{d_constant}_Lb{controller.L_base}_Lg{controller.L_gain}"
    run_dir = setup_results_dir("Fase1", track_name, param_str)

    save_metadata(run_dir,
                  params_dict={
                      "Controllo": "Pure Pursuit (Solo)",
                      "Duty_Cycle": d_constant,
                      "L_base": controller.L_base,
                      "L_gain": controller.L_gain,
                      "T_sim": T_sim
                  },
                  stats_dict={
                      "Errore_Medio": np.mean(errors),
                      "Errore_Max": np.max(errors),
                      "Errore_RMS": np.sqrt(np.mean(np.array(errors) ** 2)),
                      "vx_media": np.mean(vx_history)
                  })

    dati_plot = {
        'Velocità vx [m/s]': {'val': vx_history},
        'Velocità vy [m/s]': {'val': vy_history},
        'Velocità omega [rad/s]': {'val': omega_history},
        'Angolo Sterzo δ [rad]': {'val': delta_history},
        'Errore Laterale [m]': {'val': errors}
    }

    # Grafici principali
    plot_sim_results(run_dir, time_history, dati_plot, reference_path, trajectory_x, trajectory_y)

    # Grafico disturbi
    if disturbance is not None:
        plot_disturbance_analysis(
            run_dir, time_history, vx_history, omega_history,
            vx_disturb_history, omega_disturb_history,
            phase_name="Fase 1",
            vx_controlled=False,
            omega_controlled=False
        )

    # Video
    video_path = os.path.join(run_dir, "video_fase1.mp4")
    genera_video_animazione(reference_path, trajectory_x, trajectory_y, dt, video_path)

    print(f"\n✓ Fase 1 completata. Risultati in: {run_dir}")


if __name__ == "__main__":
    run_fase1_simulation()