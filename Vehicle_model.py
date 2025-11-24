import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from dataclasses import dataclass
import math


def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

@dataclass
class VehicleState:
    X: float  # Posizione X globale [m]
    Y: float  # Posizione Y globale [m]
    phi: float  # Orientamento [rad]
    vx: float  # Velocità longitudinale body frame [m/s]
    vy: float  # Velocità laterale body frame [m/s]
    omega: float  # Velocità angolare [rad/s]

    def to_array(self) -> np.ndarray:

        return np.array([self.X, self.Y, self.phi, self.vx, self.vy, self.omega])

    @classmethod
    def from_array(cls, arr: np.ndarray):

        return cls(X=arr[0], Y=arr[1], phi=arr[2],
                   vx=arr[3], vy=arr[4], omega=arr[5])


@dataclass
class VehicleInput:
    d: float  # Duty cycle motore [0, 1]
    delta: float  # Angolo di sterzo [rad]

    def saturate(self): #funzione per evitare che i miei parametri di input superino certi limiti
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
        self.lf = lf  #distanze dal centro di massa del veicolo asse anteriore e posteriore
        self.lr = lr

        self.Cm1 = 0.287   #$F_{motore} = d \cdot (C_{m1} - C_{m2} \cdot v_x)$. $C_{m1}$
        self.Cm2 = 0.0545


        self.Cr0 = 0.0518  # resistenza/attrito
        self.Cr2 = 0.00035

       #reattività del veicolo per arrivare all'omega target
        self.tau_omega = 0.05

    def f(self, state: VehicleState, u: VehicleInput) -> np.ndarray:   # calcolo derivate


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


        #calcolo omega_target
        eps = 1e-8
        vx_for_omega = max(abs(vx), eps) * (1 if vx >= 0 else -1)
        omega_target = (vx_for_omega / self.wb) * math.tan(delta) if abs(self.wb) > eps else 0.0
        vy_target = omega_target * self.lr


        phi_dot = omega

        # Forze longitudinali
        F_motor = d * self.Cm1 - d * self.Cm2 * vx
        # resistenza con segno opposto al movimento
        F_resistance = - self.Cr0 * np.sign(vx) - self.Cr2 * vx * abs(vx)

        # Dinamica longitudinale
        vx_dot = (F_motor + F_resistance) / self.mass


       # vy_dot = - self.k_vy * vy #cambiare tornare a modelloo dinamico professore
        vy_dot = (vy_target - vy) / max(self.tau_omega, 1e-9)

        # omega_dot: tracking verso omega_target
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
