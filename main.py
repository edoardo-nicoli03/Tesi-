import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import math



@dataclass
#RAPPRESENRAZIONE STATO VEICOLO
class VehicleState:
    X: float  #posizione veicolo sull'asse verticale
    Y: float  #posizione veicolo asse orizzontale
    phi: float #orientamento del veicolo rispetto all'asse di riferimento globale (se phi = 0 il veicolo è allineato con l'asse X, se phi è maggiore di 0 il veicolo è ruotato in sento antiorario)
    vx: float  #velocità longitudinale veicolo
    vy: float  #velocità laterale del veicolo
    omega: float #velocità angolare, velocità con cui il veicolo ruota attorno al proprio asse

    @classmethod
    def to_array(self) -> np.ndarray: #converto lo stato in un array per facilitare i calcoli (con numpy)
       return np.array([self.X, self.Y, self.phi, self.vx, self.vy, self.omega])
    def from_array(cls, arr : np.ndarray) :
     return cls(X = arr[0], Y = arr[1], phi = arr[2], vx = arr[3], vy = arr[4], omega = arr[5])

@dataclass
#COMANDI DI CONTROLLO DEL VEICOLO
class VehicleInput:
   d: float  # Duty Cycle - accelerazione del motore
   delta: float  #angolo di sterzo δ [radianti]

   def saturate(self):
       self.d = np.clip(self.d, 0.0, 1.0)
       self.delta = np.clip(self.delta, -0.35, 0.35)
       return self


class KinematicBycicleModel:

    """ Caratteristiche:
    - Ruote fisse (non si muovono, solo sterzo controllabile)
    - vy ≈ 0
    - Nessuna dinamica complessa delle forze
    - Solo equazioni geometriche del bicycle model
    Input:
    - d: duty cycle motore (ruote posteriori)
    - delta: angolo sterzo (ruote anteriori fisse)"""

    def __init__(self,
                 wheelbase: float = 0.062,
                 mass : float = 0.041,
                 Cm1: float = 0.287,
                 Cm2: float = 0.0545,
                 Cr0: float = 0.0518,
                 Cr2: float = 0.00035):
        self.wb = wheelbase #distanza tra assale anteriore e posteriore
        self.mass = mass #massa
        self.Cm1 = Cm1 #guadagno motore
        self.Cm2 = Cm2 #drag motore
        self.Cr0 = Cr0 #resistenza
        self.Cr2 = Cr2 #resistenza aerodinamica


    def f(self, state: VehicleState, input: VehicleInput) -> np.ndarray :


        X_dot = state.vx * np.cos(state.phi)
        Y_dot = state.vx * np.sin(state.phi)
        phi_dot = (state.vx / self.wb) * np.tan(input.delta)

        F_motor = input.d * self.Cm1 - input.d * self.Cm2 * state.vx

        F_resistance = -self.Cr0 - self.Cr2 * state.vx**2

        vx_dot = (F_motor + F_resistance) / self.mass
        vy_dot = 0.0
        omega_dot = 0.0

        return np.array([X_dot, Y_dot, phi_dot, vx_dot, vy_dot, omega_dot])

class DynamicBicycleModel:
         def __init__ (self, wheelbase: float = 0.062, mass: float = 0.041, inertia: float = 27.8e-6, lf: float = 0.029, lr: float = 0.033):
            self.wb = wheelbase
            self.mass = mass #massa veicolo
            self.inertia = inertia #inerzia veicolo
            self.lf = lf #distanza CG-asse anteriore
            self.lr = lr #distanza CG-asse posteriore

            self.Cm1 = 0.287
            self.Cm2 = 0.0545
            self.Cr0 = 0.0518
            self.Cr2 = 0.00035
            self.C_lat = 5.0 #coefficiente di rigidezza laterale

         def f(self, state: VehicleState, input: VehicleInput) -> np.ndarray:
            X_dot = state.vx * np.cos(state.phi) - state.vy * np.sin(state.phi)
            Y_dot = state.vx * np.sin(state.phi) + state.vy * np.cos(state.phi)
            phi_dot = state.omega

            F_motor = input.d * self.Cm1 - input.d * self.Cm2 * state.vx
            F_resistance = -self.Cr0 - self.Cr2 * state.vx**2
            F_long = F_motor + F_resistance

            F_lat = -self.C_lat * input.delta * state.vx

            M_z = F_lat * self.lf

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
class PurePursuit :


       def __init__ (self, wheelbase: float = 0.062 , lookahead_base: float = 0.15, lookahead_gain: float = 0.1, max_steering: float = 0.35, max_speed = 3.5):


           self.wb = wheelbase
           self.L_base = lookahead_base
           self.L_gain = lookahead_gain
           self.max_steering = max_steering
           self.max_speed = max_speed

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

       def pure_pursuit(self, state: VehicleState, path: List[Tuple[float, float]],
                        vx_desired : float = 1.0)-> Tuple[float, float]:



           current_speed = np.sqrt(state.vx**2 + state.vy**2)
           L = self.L_base + self.L_gain * current_speed

           lookahead_point = self._find_lookahead_point(state, path, L)

           if lookahead_point is None:
               lookahead_point = path[-1] #Se il lookahead point è nullo allora prendo l'ultimo waypoint

           dx = lookahead_point[0] - state.X  # [m]
           dy = lookahead_point[1] - state.Y  # [m]
           L_actual = np.sqrt(dx ** 2 + dy ** 2)

           target_angle  = np.arctan2(dy, dx)

           alpha = target_angle - state.phi
           alpha = self._normalize_angle(alpha)

           if L_actual > 1e-3:
               k = 2 * np.sin(alpha) / L_actual  # curvatura [1/m]
           else:
               k = 0.0

           vx_star = vx_desired  # velocità desiderata [m/s]
           omega_star = vx_star * k  # velocità angolare desiderata [rad/s]
           omega_star = np.clip (omega_star, -8.0, 8.0)

           return omega_star, vx_star

       def _find_lookahead_point(self, state: VehicleState, path: List[Tuple[float, float]],
                                 L: float) -> Tuple[float, float]:
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












 








