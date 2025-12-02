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


        self.Cr0 = 0.0518 # resistenza/attrito
        self.Cr2 = 0.00035
        self.Cf = 2
        self.Cr = 2

    def f(self, state: VehicleState, u: VehicleInput) -> np.ndarray:
            delta = np.clip(u.delta, -0.35, 0.35)
            d = np.clip(u.d, 0.0, 1.0)


            vx = state.vx
            vy = state.vy
            omega = state.omega
            phi = state.phi


            eps = 1e-4
            vx_safe = vx if abs(vx) > eps else eps * np.sign(vx) #per evitare divisione per 0


            # alpha_f = delta - atan((vy + lf*omega) / vx)
            alpha_f = delta - math.atan((vy + self.lf * omega) / vx_safe)

            # alpha_r = - atan((vy - lr*omega) / vx)
            alpha_r = - math.atan((vy - self.lr * omega) / vx_safe)

           # Calcolo delle Forze Laterali (Fy)
           #ho tolto la magic formula dal codice originale
            Fy_f = self.Cf * alpha_f
            Fy_r = self.Cr * alpha_r


            Fx = d * self.Cm1 - d * self.Cm2 * vx - self.Cr0 * np.sign(vx) - self.Cr2 * vx * abs(vx)


            #vx_dot = (Fx - Fy_f * sin(delta)) / m + vy * omega
            vx_dot = (Fx - Fy_f * math.sin(delta)) / self.mass + vy * omega

            # Accelerazione laterale
            # vy_dot = (Fy_r + Fy_f * cos(delta)) / m - vx * omega
            vy_dot = (Fy_r + Fy_f * math.cos(delta)) / self.mass - vx * omega

            # Accelerazione angolare
            # omega_dot = (Fy_f * lf * cos(delta) - Fy_r * lr) / I
            omega_dot = (Fy_f * self.lf * math.cos(delta) - Fy_r * self.lr) / self.inertia

            # Cinematica globale
            X_dot = vx * math.cos(phi) - vy * math.sin(phi)
            Y_dot = vx * math.sin(phi) + vy * math.cos(phi)
            phi_dot = omega

            return np.array([X_dot, Y_dot, phi_dot, vx_dot, vy_dot, omega_dot])





class VehicleIntegrator:
    def __init__(self, model, dt: float = 0.001):

        self.model = model
        self.dt = dt

    def Eulero(self, state: VehicleState, input: VehicleInput) -> VehicleState:

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
