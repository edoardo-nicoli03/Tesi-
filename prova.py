from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleIntegrator, VehicleState, VehicleInput

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def run_open_loop_step_test():
    dt = 0.001
    T_sim = 5.0
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

    d_step = 0.3
    delta = 0.0

    vx_history = []
    time_history = []

    for step in range(num_steps):
        t = step * dt

        control_input = VehicleInput(d=d_step, delta=delta)

        state = integrator.Eulero(state, control_input)

        vx_history.append(state.vx)
        time_history.append(t)

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, vx_history, label="v_x(t)")
    plt.title("Risposta al gradino del sistema (anello aperto)")
    plt.xlabel("Tempo [s]")
    plt.ylabel("v_x [m/s]")
    plt.grid()
    plt.legend()
    plt.show()

    return time_history, vx_history, d_step
