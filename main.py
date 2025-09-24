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
            vx_dot = F_long / self.m + state.vy * state.omega  # effetto centripeto
            vy_dot = F_lat / self.m - state.vx * state.omega  # effetto centripeto
            omega_dot = M_z / self.Iz

            return np.array([X_dot, Y_dot, phi_dot, vx_dot, vy_dot, omega_dot])




 "Discretizzazione con Eulero"
class VehicleIntegrator :
