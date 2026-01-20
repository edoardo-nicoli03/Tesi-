import matplotlib
import os

matplotlib.use('TkAgg')
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Veicolo.Vehicle_model import (VehicleIntegrator, VehicleInput, VehicleState,
                                   DynamicBicycleModel, Generator_Noise)
from Veicolo.Vehicle_controller import PurePursuit
from utils.trajectory_generator import get_trajectory
from utils.plotting_utils import (setup_results_dir, plot_sim_results,
                                  genera_video_animazione, save_metadata,
                                  plot_disturbance_analysis)


# ==============================================================
#                    SIMULAZIONE FASE 2
# ==============================================================

def run_fase2_simulation():
    dt = 0.001  # passo di integrazione
    T_sim = 40.0  # durata simulazione
    num_steps = int(T_sim / dt)

    print("=" * 70)
    print("FASE 2: PURE PURSUIT + PI LONGITUDINALE")
    print("=" * 70)

    reference_path, track_name = get_trajectory('racing')
    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    controller = PurePursuit(
        wheelbase=0.062,
        lookahead_base=0.15,
        lookahead_gain=0.12
    )

    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    trajectory_x = [state.X]
    trajectory_y = [state.Y]
    time_history = [0.0]
    vx_history = [state.vx]
    vy_history = [state.vy]
    omega_history = [state.omega]
    d_history = [0.0]
    vx_disturb_history = [0.0]
    omega_disturb_history = [0.0]

    vx_des = 1.5  # velocità desiderata

    kp = 10  # guadagno proporzionale
    omega_cross = 1
    Ti = 2 / omega_cross
    ki = kp / Ti
    error_inte = 0.0

    print(f"\n{'=' * 70}")
    print("CONFIGURAZIONE FASE 2")
    print(f"{'=' * 70}")
    print(f"Controllo laterale: Pure Pursuit → delta (DIRETTO)")
    print(f"Controllo longitudinale: PI → duty cycle")
    print(f"  - vx target: {vx_des} m/s")
    print(f"  - Kp: {kp}")
    print(f"  - Ki: {ki:.2f}")
    print(f"  - Ti: {Ti:.2f} s")
    print(f"Lookahead base: {controller.L_base} m")
    print(f"Lookahead gain: {controller.L_gain}")
    print(f"{'=' * 70}\n")


    USE_DISTURBANCE = False

    if USE_DISTURBANCE:
        disturbance = Generator_Noise(
            disturb_vx=True,
            disturb_omega=True,
            disturb_position=True,
            disturb_heading=True,
            magnitude=0.15,  # Per vx e omega
            magnitude_position=0.05,
            magnitude_heading=0.02,
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

            # Misure disturbate
            vx_measured = state.vx + disturb_vx
            omega_measured = state.omega + disturb_omega
            X_measured = state.X + disturb_X
            Y_measured = state.Y + disturb_Y
            phi_measured = state.phi + disturb_phi

            # Logging disturbi
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

        # Stato con le misure disturbate
        state_measured = VehicleState(
            X=X_measured,
            Y=Y_measured,
            phi=phi_measured,
            vx=vx_measured,
            vy=state.vy,
            omega=omega_measured
        )


        delta = controller.pure_pursuit(state_measured, reference_path)

        # ========================================
        # 3. PI LONGITUDINALE → duty cycle
        # ========================================
        error_vx = vx_des - state_measured.vx
        error_inte += error_vx * dt
        d_cmd = error_inte * ki + kp * error_vx
        d_cmd = np.clip(d_cmd, 0, 1.0)


        control_input = VehicleInput(d=d_cmd, delta=delta)
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
            print(f"t={current_time:.1f}s: pos=({state.X:.3f}, {state.Y:.3f}), "
                  f"vx={state.vx:.3f}, vy={state.vy:.4f}")

    print("\n✓ Simulazione completata!")

    # ===== STATISTICHE =====
    errors = []
    for x, y in zip(trajectory_x, trajectory_y):
        min_dist = min(np.sqrt((x - rx) ** 2 + (y - ry) ** 2) for rx, ry in reference_path)
        errors.append(min_dist)

    print("\n=========== STATISTICHE FINE SIMULAZIONE ===========")
    print(f"Errore medio: {np.mean(errors):.4f} m")
    print(f"Errore massimo: {np.max(errors):.4f} m")
    print(f"Errore quadratico medio: {np.sqrt(np.mean(np.array(errors) ** 2)):.4f} m")
    print("====================================================")

    # ===== SALVATAGGIO =====
    param_str = f"Vx{vx_des}_Kp{kp}_Ki{ki:.2f}_Ti{Ti:.2f}"
    run_dir = setup_results_dir("Fase2", track_name, param_str)

    save_metadata(run_dir,
                  params_dict={"Kp": kp, "Ki": ki, "Ti": Ti, "Vx_target": vx_des, "dt": dt, "T_sim": T_sim},
                  stats_dict={"Errore_Medio": np.mean(errors), "Errore_Max": np.max(errors),
                              "Errore_RMS": np.sqrt(np.mean(np.array(errors) ** 2))})

    dati_plot = {
        'Velocità vx [m/s]': {'val': vx_history, 'ref': vx_des},
        'Velocità vy [m/s]': {'val': vy_history},
        'Velocità omega [rad/s]': {'val': omega_history},
        'Errore Laterale [m]': {'val': errors},
        'Duty Cycle [0-1]': {'val': d_history}
    }

    # Grafici principali
    plot_sim_results(run_dir, time_history, dati_plot, reference_path, trajectory_x, trajectory_y)

    # Grafico disturbi
    if disturbance is not None:
        plot_disturbance_analysis(
            run_dir, time_history, vx_history, omega_history,
            vx_disturb_history, omega_disturb_history,
            phase_name="Fase 2",
            vx_controlled=True,
            omega_controlled=False
        )

    # Video
    video_path = os.path.join(run_dir, "simulazione_animata.mp4")
    genera_video_animazione(reference_path, trajectory_x, trajectory_y, dt, video_path)

    print(f"\n✓ Fase 2 completata! Risultati in: {run_dir}")


if __name__ == "__main__":
    run_fase2_simulation()