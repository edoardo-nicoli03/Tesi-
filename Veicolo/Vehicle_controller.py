import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from typing import Tuple, List
from Veicolo.Vehicle_model import VehicleState, normalize_angle



class PurePursuit:
    """Algoritmo Pure Pursuit per il controllo della traiettoria"""

    def __init__(self, wheelbase: float = 0.062, lookahead_base: float = 0.80,
                 lookahead_gain: float = 0.8, max_steering: float = 0.35,
                 max_speed: float = 3.5, max_yaw_rate: float = 5.0,):

        self.wb = wheelbase
        self.L_base = lookahead_base
        self.L_gain = lookahead_gain
        self.max_steering = max_steering
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.L_min = 0.10
        self.L_max = 0.40

    def pure_pursuit(self, state: VehicleState, path: List[Tuple[float, float]]) -> float:
        """
        Algoritmo Pure Pursuit - calcola l'angolo di sterzo
        """
        current_speed = np.sqrt(state.vx ** 2 + state.vy ** 2)
        L = self.L_base + self.L_gain * current_speed   #  Calcola distanza lookahead adattiva in base alla velocità current_speed
        L = np.clip(L, self.L_min, self.L_max)

        #  Trova lookahead point
        lookahead_point = self._find_lookahead_point(state, path, L)

        if lookahead_point is None:
            lookahead_point = path[-1] #se non trovo  nessun Lookahead_point ritorno a quello prima

        dx = lookahead_point[0] - state.X  #distanza x dal lookahead
        dy = lookahead_point[1] - state.Y  #distanza y dal lookahead
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


    def yaw_rate_des (self, state: VehicleState, path: List[Tuple[float, float]]) -> float:
        "Metodo per calcolare la velocità angolare desiderata"
        current_speed = np.sqrt(state.vx ** 2 + state.vy ** 2)
        L = self.L_base + self.L_gain * current_speed
        L = np.clip(L, self.L_min, self.L_max)

        lookahead_point = self._find_lookahead_point(state, path, L)
        if lookahead_point is None:
            return 0.0
        dx = lookahead_point[0] - state.X
        dy = lookahead_point[1] - state.Y
        L_actual = np.sqrt(dx ** 2 + dy ** 2)

        # 4. Calcola angolo alpha
        target_angle = np.arctan2(dy, dx)
        alpha = normalize_angle(target_angle - state.phi)

        # 5. Calcola curvatura
        if L_actual > 1e-3:
            k = 2 * np.sin(alpha) / L_actual
        else:
            k = 0.0

        # 6. Converti curvatura in velocità angolare desiderata
        # ω_des = κ · v_x
        vx_safe = max(abs(state.vx), 0.1)  # Evita divisione per zero
        omega_des = k * vx_safe

        # 7. Limita ai valori fisicamente realizzabili
        omega_des = np.clip(omega_des, -self.max_yaw_rate, self.max_yaw_rate)

        return omega_des

    def _find_lookahead_point(self, state: VehicleState,
                              path: List[Tuple[float, float]],
                              L: float) -> Tuple[float, float]:
        """
        Trova il look-ahead point sulla traiettoria

        """
        if len(path) < 2:
            return None

        vehicle_pos = np.array([state.X, state.Y]) # posizione del veicolo nello spazio

        # Trova waypoint più vicino
        distances = []
        for point in path:
            dist = np.linalg.norm(np.array(point) - vehicle_pos) #distanza tra il veicolo e ogni waypoint del percorso, calcolo la norma
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

        # se non trova l'ultimo waypoint cercato
        return path[-1]
