import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import math


 #MODELLO VEICOLO
@dataclass
class VehicleState:
    X: float  #posizione veicolo sull'asse verticale
    Y: float  #posizione veicolo asse orizzontale
    phi: float
    vx: float  #velocità longitudinale veicolo
    vy: float  #velocità laterale del veicolo
    omega: float

def to_array(self) -> np.ndarray: #converto lo stato in un array per facilitare i calcoli con numpy
    return np.array([self.X, self.Y, self.phi, self.vx, self.vy, self.omega])

def from_array(cls, arr : np.ndarray) :
    return cls(X = arr[0], Y = arr[1], phi = arr[2], v = arr[3], omega = arr[4])

@dataclass
class VehicleInput:
   d: float  # Duty Cycle - accelerazione del motore
   delta: float #angolo di sterzo δ [radianti]


class KinematicBycicleModel:

    """ Caratteristiche:
    - Ruote fisse (non si muovono, solo sterzo controllabile)
    - vy ≈ 0 (nessuna deriva laterale a velocità basse)
    - Nessuna dinamica complessa delle forze
    - Solo equazioni geometriche del bicycle model

    Vantaggi per tesi triennale:
    - Semplice da comprendere e spiegare
    - Facile da implementare e debuggare
    - Focus sui concetti Pure Pursuit
    - Risultati convincenti per velocità moderate

    Input:
    - d: duty cycle motore (ruote posteriori)
    - delta: angolo sterzo (ruote anteriori fisse)"""

    def __init__(self, wheelbase: float = 2.7):  #parametro che indica il passo del veicolo
        self.wb = wheelbase

    def f(self, state: VehicleState, input: VehicleInput) -> np.ndarray :

        K_motor = 8.0 #guadagno motore per duty cycle
        K_drag = 0.2 #coefficiente di resistenza aerodinamica
        X_dot = state.vx * np.cos(state.phi)
        Y_dot = state.vx * np.sin(state.phi)
        phi_dot = (state.vx / self.wb) * np.tan(input.delta)

        vx_dot = input.d * K_motor - K_drag * state.vx * np.cos(phi_dot)
        vy_dot = 0.0
        omega_dot = 0.0

        return np.array([X_dot, Y_dot, phi_dot, vx_dot, vy_dot, omega_dot])

class DynamicBicycleModel:
         def __init__ (self, wheelbase: float = 2.7, mass: float = 1500.0, inertia: float = 3000.0):
            self.wb = wheelbase
            self.mass = mass #massa veicolo
            self.inertia = inertia #inerzia veicolo

         def f(self, state: VehicleState, input: VehicleInput) -> np.ndarray:
            X_dot = state.vx * np.cos(state.phi) - state.vy * np.sin(state.phi)
            Y_dot = state.vx * np.sin(state.phi) + state.vy * np.cos(state.phi)
            phi_dot = state.omega
            K_motor = 8000.0
            K_drag = 400.0
            K_roll = 200.0

            F_long = input.d * K_motor - K_drag * state.vx * np.sign(state.vx)
            C_lat = 15000.0  # rigidezza pneumatico laterale [N/rad]
            F_lat = -C_lat * input.delta * (state.vx / 10.0)  # normalizzato per vx=10m/s

            # Momento di imbardata (da forza laterale sull'asse anteriore)
            l_f = self.wb * 0.6  # distanza baricentro-asse anteriore [m]
            M_z = F_lat * l_f

            # Equazioni dinamiche nel body frame
            vx_dot = F_long / self.mass + state.vy * state.omega  # effetto centripeto
            vy_dot = F_lat / self.mass - state.vx * state.omega  # effetto centripeto
            omega_dot = M_z / self.inertia

            return np.array([X_dot, Y_dot, phi_dot, vx_dot, vy_dot, omega_dot])
         




      #DISCRETIZZAZIONE CON EULERO
class VehicleIntegrator :
       def __init__ (self , model , dt: float = 0.01):  #valore del tempo di campionamento impostato a 0.01
           self.model = model
           self.dt = dt

       def Eulero (self , state : VehicleState, input: VehicleInput)-> VehicleState:
           """
           Discretizzazione con Eulero esplicita
           formula : [ x_{k+1} - x_{k} ] / dt = f(x_{k}, u_{k}
           Passi :
           1 Calcolare State_dot =  f(x_{k}, u_{k} usando modello veicolo
           2 Moltiplicare State_dot per il tempo di campionamento prescelto dt
           3 Sommarlo allo stato corrente x_{k}
           4 normalizzare l'angolo

           """

           state_dot = self.model.f(state, input) #derivate dei 6 elementi

           x_k = state.to_array() #converto lo stato attuale in array in modo da poter svolgere le operazioni

           x_k_plus1 = x_k + self.dt * state_dot #calcolo effettivo di x_{k+1} = x_k + dt * f(x_{k}, u{k})

           x_k_plus1[2]= self._normalize_angle(x_k_plus1[2]) #normalizzo l'angolo tra pigreco e -pigreco, è un array quindi sto cambiando il secondo elemento della variabile (l'angolo phi)

           return VehicleState.from_array(x_k_plus1)


      #IMPLEMENTAZIONE PUREPURSUIT

       def __init__ (self, wheelbase: float = 2.7 , lookahead_base: float = 3.0, lookahead_gain: float = 0.2):


           self.wb = wheelbase
           self.L_base = lookahead_base
           self.L_gain = lookahead_gain

       def pure_pursuit(self, state: VehicleState, path: List[Tuple[float, float]],
                        vx_desired : float = 8.0)-> Tuple[float, float]:

           """Algoritmo Pure Pursuit principale

        Passo-passo:
        1. Calcola distanza lookahead L adattiva
        2. Trova look-ahead point sulla traiettoria
        3. Calcola angolo alpha tra heading e vettore verso look-ahead
        4. Calcola curvatura desiderata k usando formula Pure Pursuit
        5. Converte in omega_star = vx_star * k
        6. Ritorna omega_star, vx_star per i PID

        Args:
            state: stato corrente del veicolo
            path: lista di waypoint [(x1,y1), (x2,y2), ...]
            vx_desired: velocità longitudinale desiderata [m/s]

        Returns:
            omega_star: velocità angolare di riferimento [rad/s]
            vx_star: velocità longitudinale di riferimento [m/s]
        """

           current_speed = np.sqrt(state.vx**2 + state.vy**2)
           L = self.L_base + self.L_gain * current_speed

           lookahead_point = self._find_lookahead_point(state, path, L)

           if lookahead_point is None:
               lookahead_point = path[-1] #Se il lookahead poin è nullo allora prendo l'ultimo waypoint

           dx = lookahead_point[0] - state.X  # [m]
           dy = lookahead_point[1] - state.Y  # [m]
           L_actual = np.sqrt(dx ** 2 + dy ** 2)

           target_angle  = np.arctan2(dy, dx)
           omega_star = np.arctan2(dy, dx)

           alpha = target_angle - state.phi
           alpha = self._normalize_angle(alpha)

           if L_actual > 1e-3:  # evita divisione per zero
               k = 2 * np.sin(alpha) / L_actual  # curvatura [1/m]
           else:
               k = 0.0

           vx_star = vx_desired  # velocità desiderata [m/s]
           omega_star = vx_star * k  # velocità angolare desiderata [rad/s]

           return omega_star, vx_star

       def _find_lookahead_point(self, state: VehicleState, path: List[Tuple[float, float]],
                                 L: float) -> Optional[Tuple[float, float]]:
           """
           Trova il look-ahead point sulla traiettoria

           Algoritmo:
           1. Trova il waypoint più vicino al veicolo
           2. Avanza lungo la traiettoria fino a trovare punto a distanza ≥ L
           3. Interpola linearmente per ottenere esattamente distanza L

           Args:
               state: stato corrente del veicolo
               path: lista waypoint
               L: distanza lookahead desiderata [m]

           Returns:
               lookahead_point: coordinate (x,y) del punto [m] o None
           """
           if len(path) < 2:
               return None

           vehicle_pos = np.array([state.X, state.Y])  # [m]

           # === Trova waypoint più vicino ===
           distances = []
           for point in path:
               dist = np.linalg.norm(np.array(point) - vehicle_pos)
               distances.append(dist)

           closest_idx = np.argmin(distances)  # indice punto più vicino

           # === Cerca punto a distanza L ===
           for i in range(closest_idx, len(path)):
               point = np.array(path[i])
               distance = np.linalg.norm(point - vehicle_pos)

               if distance >= L:
                   if i > closest_idx:
                       # Interpola tra path[i-1] e path[i]
                       prev_point = np.array(path[i - 1])
                       prev_distance = np.linalg.norm(prev_point - vehicle_pos)

                       # Interpolazione lineare per ottenere distanza esatta L
                       if distance > prev_distance:  # evita divisione per zero
                           alpha = (L - prev_distance) / (distance - prev_distance)
                           alpha = np.clip(alpha, 0.0, 1.0)  # limita in [0,1]

                           interpolated_point = prev_point + alpha * (point - prev_point)
                           return (interpolated_point[0], interpolated_point[1])

                   # Se non serve interpolazione, usa il punto direttamente
                   return (point[0], point[1])

           # Non trovato: usa ultimo punto del path
           return path[-1]

       @staticmethod
       def _normalize_angle(angle: float) -> float:
           """Normalizza angolo in [-π, π]"""
           while angle > np.pi:
               angle -= 2 * np.pi
           while angle < -np.pi:
               angle += 2 * np.pi
           return angle












 








