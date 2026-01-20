import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Veicolo.Vehicle_model import (VehicleIntegrator, VehicleInput,
                                   VehicleState, DynamicBicycleModel,
                                   Generator_Noise)
from Veicolo.Vehicle_controller import PurePursuit
from Veicolo.PID_controller import YawRatePIDController, VelocityPIDController
from utils.trajectory_generator import get_trajectory
from utils.plotting_utils import (setup_results_dir, plot_sim_results,
                                  genera_video_animazione, save_metadata,
                                  plot_disturbance_analysis)


# ==============================================================
#                    SIMULAZIONE PRINCIPALE
# ==============================================================

def run_fase3_simulation():


    dt = 0.001  # Passo di integrazione [s]
    T_sim = 40.0  # Durata simulazione [s]
    num_steps = int(T_sim / dt)

    print("=" * 70)
    print("SIMULAZIONE CON CONTROLLO IN CASCATA")
    print("Pure Pursuit -> PID laterale (ω) -> δ")
    print("=" * 70)

    reference_path, track_name = get_trajectory('racing')


    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

   #Pure Pursuit
    pure_pursuit = PurePursuit(
        wheelbase=0.062,
        lookahead_base=0.15,
        lookahead_gain=0.12,
        max_yaw_rate=5.0
    )

    vx_des = 1.5  # Velocità desiderata [m/s]

    pid_longitudinal = VelocityPIDController(
        kp=10.0,
        ki=5.0,
        dt=dt,
        max_duty=1.0
    )



    # Solo P
    kp_omega = 0.3
    ki_omega = 0.0
    kd_omega = 0.0
    use_integral = False
    use_derivative = False


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

    # ===== CONFIGURAZIONE DISTURBI =====
    USE_DISTURBANCE = True

    if USE_DISTURBANCE:
        disturbance = Generator_Noise(
            disturb_vx=False,
            disturb_omega=False,
            disturb_position=True,
            disturb_heading=True,
            magnitude=0.15,
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

    # ===== STATO INIZIALE =====
    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)


    trajectory_x = [state.X]
    trajectory_y = [state.Y]
    time_history = [0.0]
    vx_history = [state.vx]
    vy_history = [state.vy]
    omega_history = [state.omega]
    omega_des_history = [0.0]
    delta_history = [0.0]
    d_history = [0.0]
    vx_disturb_history = [0.0]
    omega_disturb_history = [0.0]

    # ===== LOOP DI SIMULAZIONE =====
    print(f"Inizio simulazione: {num_steps} step, dt={dt}s")
    print("Questo potrebbe richiedere qualche secondo...\n")

    for step in range(num_steps):
        current_time = step * dt


        if disturbance is not None:
            # Disturbi su velocità
            disturb_vx = disturbance.get_disturbance(current_time, 'vx')
            disturb_omega = disturbance.get_disturbance(current_time, 'omega')


            disturb_X = disturbance.get_disturbance(current_time, 'position')
            disturb_Y = disturbance.get_disturbance(current_time, 'position')
            disturb_phi = disturbance.get_disturbance(current_time, 'heading')

            # Misure disturbate
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

        # Stato con TUTTE le misure disturbate
        state_measured = VehicleState(
            X=X_measured,  # ← DISTURBATO
            Y=Y_measured,  # ← DISTURBATO
            phi=phi_measured,  # ← DISTURBATO
            vx=vx_measured,
            vy=state.vy,
            omega=omega_measured
        )

        # ========================================
        # 2. PURE PURSUIT → omega_des (usa stato disturbato)
        # ========================================
        omega_des = pure_pursuit.yaw_rate_des(state_measured, reference_path)

        # ========================================
        # 3. PID LATERALE: omega → delta
        # ========================================
        delta = pid_lateral.compute(omega_des, state_measured.omega)

        # ========================================
        # 4. PID LONGITUDINALE: vx → d
        # ========================================
        d_cmd = pid_longitudinal.compute(vx_des, state_measured.vx)

        # ========================================
        # 5. APPLICA CONTROLLO AL VEICOLO
        # ========================================
        control_input = VehicleInput(d=d_cmd, delta=delta)
        control_input.saturate()


        state = integrator.Eulero(state, control_input)


        trajectory_x.append(state.X)
        trajectory_y.append(state.Y)
        time_history.append(current_time + dt)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        omega_history.append(state.omega)
        omega_des_history.append(omega_des)
        delta_history.append(delta)
        d_history.append(d_cmd)


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

    # ===== SALVATAGGIO =====
    param_str = f"Cascata_{control_type}_Kp{kp_omega:.2f}_Vx{vx_des}"
    run_dir = setup_results_dir("Fase3", track_name, param_str)

    save_metadata(run_dir,
                  params_dict={
                      "Controllo": f"Cascata ({control_type})",
                      "Vx_target": vx_des,
                      "PID_Long_Kp_Ki": [pid_longitudinal.kp, pid_longitudinal.ki],
                      "PID_Lat_Kp_Ki_Kd": [pid_lateral.kp, pid_lateral.ki, pid_lateral.kd],
                      "L_base": pure_pursuit.L_base,
                      "T_sim": T_sim
                  },
                  stats_dict={
                      "Errore_Medio_m": np.mean(errors),
                      "Errore_Max_m": np.max(errors),
                      "Errore_RMS_m": np.sqrt(np.mean(np.array(errors) ** 2))
                  })

    dati_plot = {
        'Velocità vx [m/s]': {'val': vx_history, 'ref': vx_des},
        'Velocità angolare ω [rad/s]': {'val': omega_history, 'ref': omega_des_history},
        'Angolo Sterzo δ [rad]': {'val': delta_history},
        'Errore Laterale [m]': {'val': errors},
        'Comando Motore (d)': {'val': d_history}
    }

    # Grafici principali
    plot_sim_results(run_dir, time_history, dati_plot, reference_path, trajectory_x, trajectory_y)

    # Grafico disturbi (se attivi)
    if disturbance is not None:
        plot_disturbance_analysis(
            run_dir, time_history, vx_history, omega_history,
            vx_disturb_history, omega_disturb_history,
            phase_name="Fase 3",
            vx_controlled=True,
            omega_controlled=True
        )

    # Video
    video_path = os.path.join(run_dir, "video_cascata.mp4")
    genera_video_animazione(reference_path, trajectory_x, trajectory_y, dt, video_path)

    print(f"\n✓ Fase 3 completata con successo. Risultati salvati in: {run_dir}")


if __name__ == "__main__":
    run_fase3_simulation()