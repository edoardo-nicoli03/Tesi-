import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import math

"""Funzione per normalizzare un angolo nell'intervallo   [-π, π]"""
def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


@dataclass
class VehicleState:
    """Rappresentazione dello stato del veicolo"""
    X: float  # Posizione X globale [m]
    Y: float  # Posizione Y globale [m]
    phi: float  # Orientamento [rad]
    vx: float  # Velocità longitudinale body frame [m/s]
    vy: float  # Velocità laterale body frame [m/s]
    omega: float  # Velocità angolare [rad/s]

    def to_array(self) -> np.ndarray:
        """Converte lo stato in array numpy"""
        return np.array([self.X, self.Y, self.phi, self.vx, self.vy, self.omega])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Crea uno stato da un array numpy"""
        return cls(X=arr[0], Y=arr[1], phi=arr[2],
                   vx=arr[3], vy=arr[4], omega=arr[5])


@dataclass
class VehicleInput:
    """Input di controllo del veicolo"""
    d: float  # Duty cycle motore [0, 1]
    delta: float  # Angolo di sterzo [rad]

    def saturate(self):
        """Satura gli input ai limiti fisici"""
        self.d = np.clip(self.d, 0.0, 1.0)
        self.delta = np.clip(self.delta, -0.35, 0.35)
        return self


class DynamicBicycleModel:

    def __init__(self,
                 wheelbase: float = 0.062,
                 mass: float = 0.041,
                 inertia: float = 27.8e-6,
                 lf: float = 0.029,
                 lr: float = 0.033):
        self.wb = wheelbase
        self.mass = mass
        self.inertia = inertia
        self.lf = lf
        self.lr = lr

        # motore
        self.Cm1 = 0.287
        self.Cm2 = 0.0545

        # resistenza/attrito
        self.Cr0 = 0.0518
        self.Cr2 = 0.00035

        # guadagno per "schiacciare" v_y a zero (più alto -> vy torna a zero più rapidamente)
        self.k_vy = 10.0

       #reattività del veicolo per arrivare all'omega target
        self.tau_omega = 0.05

    def f(self, state: VehicleState, u: VehicleInput) -> np.ndarray:

        # saturazione di sicurezza
        delta = np.clip(u.delta, -0.35, 0.35)
        d = np.clip(u.d, 0.0, 1.0)

        X = state.X
        Y = state.Y
        phi = state.phi
        vx = state.vx
        vy = state.vy
        omega = state.omega

        #velocità del veicolo sui due assi cartesiani
        X_dot = vx * math.cos(phi) - vy * math.sin(phi)
        Y_dot = vx * math.sin(phi) + vy * math.cos(phi)


      #calcolo omega_target con protezione per eventuali errori
        eps = 1e-8
        vx_for_omega = max(abs(vx), eps) * (1 if vx >= 0 else -1)
        omega_target = (vx_for_omega / self.wb) * math.tan(delta) if abs(self.wb) > eps else 0.0


        phi_dot = omega

        # Forze longitudinali
        F_motor = d * self.Cm1 - d * self.Cm2 * vx
        # resistenza con segno (opposta al movimento)
        F_resistance = - self.Cr0 * np.sign(vx) - self.Cr2 * vx * abs(vx)

        # Dinamica longitudinale
        vx_dot = (F_motor + F_resistance) / self.mass


        vy_dot = - self.k_vy * vy

        # omega_dot: tracking verso omega_target con primo ordine (tau piccolo)
        omega_dot = (omega_target - omega) / max(self.tau_omega, 1e-9)

        return np.array([X_dot, Y_dot, phi_dot, vx_dot, vy_dot, omega_dot])


class VehicleIntegrator:
    def __init__(self, model, dt: float = 0.01):

        self.model = model
        self.dt = dt

    def Eulero(self, state: VehicleState, input: VehicleInput) -> VehicleState:
        """
        Discretizzazione con metodo di Eulero esplicito
        Formula: x_{k+1} = x_k + dt * f(x_k, u_k)
        """
        # Calcola le derivate
        state_dot = self.model.f(state, input)

        # Converte stato in array
        x_k = state.to_array()

        # Integrazione
        x_k_plus1 = x_k + self.dt * state_dot

        # Normalizza angolo
        x_k_plus1[2] = normalize_angle(x_k_plus1[2])

        #  Converte in stato
        return VehicleState.from_array(x_k_plus1)


class PurePursuit:
    """Algoritmo Pure Pursuit per il controllo della traiettoria"""

    def __init__(self, wheelbase: float = 0.062, lookahead_base: float = 0.15,
                 lookahead_gain: float = 0.1, max_steering: float = 0.35,
                 max_speed: float = 3.5):
        """
        Inizializza il controllore Pure Pursuit

        Args:
            wheelbase: distanza tra gli assi [m]
            lookahead_base: distanza lookahead base [m]
            lookahead_gain: guadagno adattivo lookahead [s]
            max_steering: massimo angolo di sterzo [rad]
            max_speed: velocità massima [m/s]
        """
        self.wb = wheelbase
        self.L_base = lookahead_base
        self.L_gain = lookahead_gain
        self.max_steering = max_steering
        self.max_speed = max_speed

    def pure_pursuit(self, state: VehicleState, path: List[Tuple[float, float]]) -> float:
        """
        Algoritmo Pure Pursuit - calcola l'angolo di sterzo
        """
        #  Calcola distanza lookahead adattiva
        current_speed = np.sqrt(state.vx ** 2 + state.vy ** 2)
        L = self.L_base + self.L_gain * current_speed

        #  Trova lookahead point
        lookahead_point = self._find_lookahead_point(state, path, L)

        if lookahead_point is None:
            lookahead_point = path[-1]

        #  Calcola vettore verso lookahead
        dx = lookahead_point[0] - state.X #distanza x dal lookahead
        dy = lookahead_point[1] - state.Y #distanza y dal lookahead
        L_actual = np.sqrt(dx ** 2 + dy ** 2)

        #  Calcola angolo alpha se alpha è positivo il punto mira a sx sennò a dx
        target_angle = np.arctan2(dy, dx)
        alpha = target_angle - state.phi
        alpha = normalize_angle(alpha)

        #  Calcola curvatura desiderata
        if L_actual > 1e-3:
            k = 2 * np.sin(alpha) / L_actual
        else:
            k = 0.0

        #  Converte curvatura in angolo di sterzo delta
        delta = np.arctan(k * self.wb)
        delta = np.clip(delta, -self.max_steering, self.max_steering) #l'angolo di sterzo deve rispettare questi limiti

        return delta
    def _find_lookahead_point(self, state: VehicleState,
                              path: List[Tuple[float, float]],
                              L: float) -> Tuple[float, float]:
        """
        Trova il look-ahead point sulla traiettoria

        Algoritmo:
        1. Trova il waypoint più vicino al veicolo
        2. Avanza lungo la traiettoria fino a trovare punto a distanza ≥ L
        3. Interpola linearmente per ottenere esattamente distanza L
        """
        if len(path) < 2:
            return None

        vehicle_pos = np.array([state.X, state.Y]) # posizione del veicolo nello spazio

        # Trova waypoint più vicino
        distances = []
        for point in path:
            dist = np.linalg.norm(np.array(point) - vehicle_pos) #distanza tra il veicolo e ogni waypoint del percorso
            distances.append(dist) #aggiungo alla lista il valore appena calcolato

        closest_idx = np.argmin(distances) #prendo l'argomento minimo della mia lista

        # Cerca punto a distanza L
        for i in range(closest_idx, len(path)):
            point = np.array(path[i])
            distance = np.linalg.norm(point - vehicle_pos)

            if distance >= L:
                if i > closest_idx:
                    # Interpola tra path[i-1] e path[i]
                    prev_point = np.array(path[i - 1])
                    prev_distance = np.linalg.norm(prev_point - vehicle_pos)

                    if distance > prev_distance:
                        alpha = (L - prev_distance) / (distance - prev_distance)
                        alpha = np.clip(alpha, 0.0, 1.0)

                        interpolated_point = prev_point + alpha * (point - prev_point)
                        return (interpolated_point[0], interpolated_point[1])

                return (point[0], point[1])

        # Non trovato: usa ultimo punto
        return path[-1]




# ============================================================================
# SCRIPT DI SIMULAZIONE
# ============================================================================

""" GENERA UNA TRAIETTORIA A FORMA DI CERCHIO"""
def generate_circular_path(radius: float, center: Tuple[float, float],
                           num_points: int = 150) -> List[Tuple[float, float]]:
    
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points)

    path = []
    for angle in angles:
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        path.append((x, y))


    return path


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


def run_simulation():

    # PARAMETRI DI SIMULAZIONE
    dt = 0.01  # tempo di campionamento [s]
    T_sim = 20.0  # durata simulazione [s]
    num_steps = int(T_sim / dt)

    # Duty cycle costante
    d_constant = 0.25

  # GENERA TRAIETTORIA

    radius = 2.0  # [m]
    center = (2.0, 0.0)  # centro in (2, 0) -> parte da (0, 0)
    reference_path = generate_circular_path(radius, center, num_points=150) #Per cerchio
   # reference_path = generate_figure_eight_path(width=4.0, num_points=200)  #Per otto
    #reference_path = generate_straight_and_turn_path(straight_len=3.0, turn_radius=1.5, num_points=200)


    #print(f"Traiettoria generata: cerchio con raggio {radius}m, centro {center}")
    print(f"Numero di waypoints: {len(reference_path)}")

    # ========================================================================
    # INIZIALIZZA MODELLO E CONTROLLORE

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    controller = PurePursuit(wheelbase=0.062)
    state = VehicleState(X=0.0, Y=0.0, phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    # ========================================================================
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
        input = VehicleInput(d=d_constant, delta=delta)
        input.saturate() #verifico che rispettino i valori di saturazione

        # Integrazione numerica con Eulero
        state = integrator.Eulero(state, input)

        trajectory_x.append(state.X)
        trajectory_y.append(state.Y)
        time_history.append(current_time + dt)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        omega_history.append(state.omega)

        # Stampa progresso
        if step % int(2.0 / dt) == 0:
            print(f"t={current_time:.1f}s: pos=({state.X:.3f}, {state.Y:.3f}), "
                  f"vx={state.vx:.3f}, vy={state.vy:.4f}")

    print("\nSimulazione completata!")

    # ========================================================================
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

    # ========================================================================
    # STATISTICHE FINALI

    print("\n" + "=" * 60)
    print("STATISTICHE SIMULAZIONE")
    print("=" * 60)
    print(f"Velocità longitudinale media: {np.mean(vx_history):.3f} m/s")
    print(f"Velocità longitudinale finale: {vx_history[-1]:.3f} m/s")
    print(f"Velocità laterale max (abs): {np.max(np.abs(vy_history)):.5f} m/s")
    print(f"Velocità laterale RMS: {np.sqrt(np.mean(np.array(vy_history) ** 2)):.5f} m/s")
    print(f"Velocità angolare max (abs): {np.max(np.abs(omega_history)):.3f} rad/s")

    # Calcola errore rispetto alla traiettoria
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