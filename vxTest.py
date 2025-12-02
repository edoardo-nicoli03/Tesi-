
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

# TUNING KP SU VX
def run_vx_test():
        dt = 0.01
        T_sim = 10.0
        num_steps = int(T_sim / dt)

        model = DynamicBicycleModel()
        integrator = VehicleIntegrator(model, dt)

        state = VehicleState(
            X=0.0, Y=0.0,
            phi=0.0,
            vx=0.1,
            vy=0.0,
            omega=0.0
        )

        vx_des = 0.5
        kp = 32

        time_history = []
        vx_history = []
        error_history = []
        d_history = []

        print("\n=== INIZIO TEST LONGITUDINALE SU v_x ===")

        for step in range(num_steps):
            t = step * dt

            error_vx = vx_des - state.vx #definizione dell'errore di velocità

            d_cmd = kp * error_vx #calcolo il duty cicle con kp (guadagno) e l'errore della velocità
            d_cmd = np.clip(d_cmd, 0.0, 1.0)

            delta_cmd = 0.0 #pongo delta = 0 per isolare il mio studio dalla traiettoria

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


        fig.savefig(os.path.join(save_dir, "grafici_vx_error_duty.png"))
        print("→ grafici_vx_error_duty.png salvato")

        with open(os.path.join(save_dir, "log.txt"), "w") as f:
            f.write(f"Test controllo v_x\n")
            f.write(f"kp = {kp}\n")
            f.write(f"vx_des = {vx_des}\n")
            f.write(f"durata sim = {T_sim}s\n")
            f.write(f"dt = {dt}\n")
            f.write(f"vx finale = {vx_history[-1]}\n")

        print("→ log.txt salvato")
        plt.show()


