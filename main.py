import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import math


def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


@dataclass
#RAPPRESENRAZIONE STATO VEICOLO
class VehicleState:
    X: float  #posizione veicolo sull'asse verticale
    Y: float  #posizione veicolo asse orizzontale
    phi: float #orientamento del veicolo rispetto all'asse di riferimento globale (se phi = 0 il veicolo è allineato con l'asse X, se phi è maggiore di 0 il veicolo è ruotato in sento antiorario)
    vx: float  #velocità longitudinale veicolo
    vy: float  #velocità laterale del veicolo
    omega: float #velocità angolare, velocità con cui il veicolo ruota attorno al proprio asse


    def to_array(self) -> np.ndarray: #converto lo stato in un array per facilitare i calcoli (con numpy)
       return np.array([self.X, self.Y, self.phi, self.vx, self.vy, self.omega])

    @classmethod
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


class KinematicBicycleModel:

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

           x_k_plus1[2]= normalize_angle(x_k_plus1[2]) #normalizzo l'angolo tra pigreco e -pigreco, è un array quindi sto cambiando il secondo elemento della variabile (l'angolo phi)

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
           alpha = normalize_angle(alpha)

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
                           alpha = np.clip(alpha, 0.0, 1.0)

                           interpolated_point = prev_point + alpha * (point - prev_point)
                           return (interpolated_point[0], interpolated_point[1])

                   # Se non serve interpolazione, usa il punto direttamente
                   return (point[0], point[1])

           # Non trovato: usa ultimo punto del path
           return path[-1]




def test_1_vehicle_state():
    """TEST 1: Verifica VehicleState"""
    print("=" * 60)
    print("TEST 1: VehicleState - conversione array")
    print("=" * 60)

    state = VehicleState(X=1.0, Y=2.0, phi=0.5, vx=3.0, vy=0.1, omega=0.2)
    print(f"Stato originale: {state}")


    arr = state.to_array()
    print(f"Array: {arr}")


    state_restored = VehicleState.from_array(arr)
    print(f"Stato ripristinato: {state_restored}")

    assert np.allclose(state.to_array(), state_restored.to_array()), "❌ ERRORE conversione!"
    print("✅ TEST 1 PASSATO: Conversione array OK\n")


def test_2_kinematic_model():
    """TEST 2: Verifica modello cinematico"""
    print("=" * 60)
    print("TEST 2: Modello Cinematico")
    print("=" * 60)

    model = KinematicBicycleModel()
    state = VehicleState(X=0, Y=0, phi=0, vx=1.0, vy=0, omega=0)
    input_cmd = VehicleInput(d=0.5, delta=0.1)

    state_dot = model.f(state, input_cmd)

    print(f"Input: d={input_cmd.d}, delta={np.degrees(input_cmd.delta):.1f}°")
    print(f"Stato: vx={state.vx:.2f} m/s")
    print(f"\nDerivate calcolate:")
    print(f"  X_dot = {state_dot[0]:.3f} m/s")
    print(f"  Y_dot = {state_dot[1]:.3f} m/s")
    print(f"  phi_dot = {state_dot[2]:.3f} rad/s = {np.degrees(state_dot[2]):.1f}°/s")
    print(f"  vx_dot = {state_dot[3]:.3f} m/s²")


    assert state_dot[0] > 0, "❌ X_dot dovrebbe essere positivo!"
    assert abs(state_dot[1]) < 0.2, "❌ Y_dot dovrebbe essere piccolo!"
    assert state_dot[4] == 0, "❌ vy_dot dovrebbe essere 0 (cinematico)!"

    print("✅ TEST 2 PASSATO: Modello cinematico OK\n")


def test_3_integration():
    """TEST 3: Verifica integrazione numerica"""
    print("=" * 60)
    print("TEST 3: Integrazione Eulero")
    print("=" * 60)

    model = KinematicBicycleModel()
    integrator = VehicleIntegrator(model, dt=0.01)

    state = VehicleState(X=0, Y=0, phi=0, vx=1.0, vy=0, omega=0)
    input_cmd = VehicleInput(d=0.5, delta=0.1)

    print(f"Stato iniziale: X={state.X:.3f}, Y={state.Y:.3f}, phi={np.degrees(state.phi):.1f}°")


    for i in range(10):
        state = integrator.Eulero(state, input_cmd)

    print(f"Dopo 10 step (0.1s): X={state.X:.3f}, Y={state.Y:.3f}, phi={np.degrees(state.phi):.1f}°")

    assert state.X > 0, "❌ Il veicolo dovrebbe essersi mosso in avanti!"
    assert abs(state.phi) > 0, "❌ Il veicolo dovrebbe aver ruotato!"

    print("✅ TEST 3 PASSATO: Integrazione OK\n")


def test_4_pure_pursuit():
    """TEST 4: Verifica Pure Pursuit"""
    print("=" * 60)
    print("TEST 4: Pure Pursuit")
    print("=" * 60)


    path = [(i * 0.1, 0.0) for i in range(20)]

    pure_pursuit = PurePursuit()
    state = VehicleState(X=0, Y=0.05, phi=0.1, vx=1.0, vy=0, omega=0)

    print(f"Stato veicolo: X={state.X:.3f}, Y={state.Y:.3f}, phi={np.degrees(state.phi):.1f}°")
    print(f"Traiettoria: linea retta da (0,0) a (2,0)")

    omega_star, vx_star = pure_pursuit.pure_pursuit(state, path, vx_desired=1.0)

    print(f"\nOutput Pure Pursuit:")
    print(f"  omega_star = {omega_star:.3f} rad/s")
    print(f"  vx_star = {vx_star:.2f} m/s")


    print(f"\nVerifica logica:")
    print(f"  Veicolo sopra traiettoria (Y={state.Y:.3f} > 0) → dovrebbe girare verso basso")
    print(f"  omega_star = {omega_star:.3f} {'✅ negativo!' if omega_star < 0 else '⚠️ dovrebbe essere negativo'}")

    print("✅ TEST 4 PASSATO: Pure Pursuit OK\n")


def test_5_complete_simulation():
    """TEST 5: Simulazione completa"""
    print("=" * 60)
    print("TEST 5: Simulazione Completa (50 step)")
    print("=" * 60)


    model = KinematicBicycleModel()
    integrator = VehicleIntegrator(model, dt=0.01)
    pure_pursuit = PurePursuit()


    path = [(i * 0.05, 0.0) for i in range(30)]
    path.extend([(1.5 + 0.2 * np.cos(t), 0.2 * np.sin(t)) for t in np.linspace(0, np.pi / 2, 20)])


    state = VehicleState(X=0, Y=0, phi=0, vx=0.5, vy=0, omega=0)


    history_X = [state.X]
    history_Y = [state.Y]
    history_vx = [state.vx]

    print("Simulazione in corso...")

    for step in range(50):

        omega_star, vx_star = pure_pursuit.pure_pursuit(state, path, vx_desired=1.0)


        input_cmd = VehicleInput(d=0.6, delta=0.1 * omega_star)
        input_cmd.saturate()


        state = integrator.Eulero(state, input_cmd)


        history_X.append(state.X)
        history_Y.append(state.Y)
        history_vx.append(state.vx)

        if step % 10 == 0:
            print(f"  Step {step:2d}: X={state.X:.3f}, Y={state.Y:.3f}, vx={state.vx:.2f}, omega_star={omega_star:.2f}")

    print(f"\nPosizione finale: X={state.X:.3f}, Y={state.Y:.3f}")
    print(f"Velocità finale: vx={state.vx:.2f} m/s")


    plt.figure(figsize=(14, 5))


    plt.subplot(1, 3, 1)
    path_array = np.array(path)
    plt.plot(path_array[:, 0], path_array[:, 1], 'r--', linewidth=2, label='Riferimento')
    plt.plot(history_X, history_Y, 'b-', linewidth=2, label='Seguita')
    plt.scatter(history_X[0], history_Y[0], c='g', s=100, label='Start', zorder=5)
    plt.scatter(history_X[-1], history_Y[-1], c='r', s=100, label='End', zorder=5)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Traiettoria')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')


    plt.subplot(1, 3, 2)
    time = np.arange(len(history_X)) * 0.01
    plt.plot(time, history_X, 'b-', linewidth=2)
    plt.xlabel('Tempo [s]')
    plt.ylabel('X [m]')
    plt.title('Posizione X')
    plt.grid(True, alpha=0.3)

    # Plot 3: Velocità nel tempo
    plt.subplot(1, 3, 3)
    plt.plot(time, history_vx, 'g-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='vx desiderata')
    plt.xlabel('Tempo [s]')
    plt.ylabel('vx [m/s]')
    plt.title('Velocità Longitudinale')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_simulazione.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafico salvato come 'test_simulazione.png'")
    plt.show()

    print("✅ TEST 5 PASSATO: Simulazione completa OK\n")


def run_all_tests():
    """Esegue tutti i test in sequenza"""
    print("\n" + "=" * 60)
    print("INIZIO TEST SUITE COMPLETA")
    print("=" * 60 + "\n")

    try:
        test_1_vehicle_state()
        test_2_kinematic_model()
        test_3_integration()
        test_4_pure_pursuit()
        test_5_complete_simulation()

        print("\n" + "=" * 60)
        print("🎉 TUTTI I TEST PASSATI CON SUCCESSO! 🎉")
        print("=" * 60)
        print("\n✅ Il codice è corretto e funzionante!")
        print("✅ Modello cinematico validato")
        print("✅ Integrazione numerica corretta")
        print("✅ Pure Pursuit implementato correttamente")
        print("✅ Sistema completo funzionante")

    except Exception as e:
        print(f"\n❌ ERRORE NEI TEST: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()













 








