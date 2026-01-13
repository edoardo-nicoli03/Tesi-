"""
Controllori PID per il sistema di controllo del veicolo autonomo.

Questo modulo implementa controllori PID configurabili per:
- Controllo laterale (velocità angolare omega → angolo sterzo delta)
- Controllo longitudinale (velocità vx → duty cycle d)

Il tuning è progressivo: si può iniziare con solo P, poi aggiungere I e D.
"""

import numpy as np


class YawRatePIDController:
    """
    Controllore PID per inseguimento della velocità angolare (yaw rate).

    FUNZIONAMENTO:
    - Riceve omega_desiderato (dal Pure Pursuit)
    - Misura omega_attuale (dal veicolo)
    - Calcola errore: e = omega_des - omega_actual
    - Genera comando: delta = Kp·e + Ki·∫e·dt + Kd·de/dt

    TUNING PROGRESSIVO:
    1. Parti con solo P (use_integral=False, use_derivative=False)
    2. Aggiungi I se c'è errore stazionario (use_integral=True)
    3. Aggiungi D se oscilla troppo (use_derivative=True)
    """

    def __init__(self,
                 kp: float = 0.3,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 dt: float = 0.001,
                 max_delta: float = 0.35,
                 use_integral: bool = False,
                 use_derivative: bool = False):
        """
        Inizializza il controllore PID per velocità angolare.

        Args:
            kp: guadagno proporzionale
                - Più alto → risposta più rapida, rischio oscillazioni
                - Più basso → risposta lenta ma stabile
                - Valori tipici: 0.1 - 1.0

            ki: guadagno integrale
                - Elimina errore a regime
                - Troppo alto → overshoot e instabilità
                - Valori tipici: 0.001 - 0.1

            kd: guadagno derivativo
                - Smorza oscillazioni
                - Anticipa variazioni
                - Troppo alto → amplifica rumore
                - Valori tipici: 0.01 - 0.2

            dt: passo di campionamento [s]
            max_delta: limite angolo sterzo [rad] (default ±0.35 rad ≈ ±20°)
            use_integral: True per attivare azione integrale
            use_derivative: True per attivare azione derivativa
        """
        self.kp = kp
        self.ki = ki if use_integral else 0.0
        self.kd = kd if use_derivative else 0.0
        self.dt = dt
        self.max_delta = max_delta

        self.use_integral = use_integral
        self.use_derivative = use_derivative

        # Stati interni del controllore
        self.integral = 0.0  # Accumulo dell'errore
        self.prev_error = 0.0  # Errore al passo precedente

        # Limiti anti-windup per l'integrale
        self.integral_max = max_delta  # Limita accumulo integrale

    def compute(self, omega_des: float, omega_actual: float) -> float:
        """
        Calcola l'angolo di sterzo per inseguire omega_des.

        FORMULA COMPLETA:
        delta = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de(t)/dt

        dove:
        - e(t) = omega_des - omega_actual (errore)
        - ∫e(τ)dτ ≈ Σ(e_k · Δt) (integrale discreto)
        - de(t)/dt ≈ (e_k - e_{k-1}) / Δt (derivata discreta)

        Args:
            omega_des: velocità angolare desiderata [rad/s]
            omega_actual: velocità angolare misurata [rad/s]

        Returns:
            delta: angolo di sterzo [rad], saturato in [-max_delta, +max_delta]
        """
        # 1. CALCOLA ERRORE
        error = omega_des - omega_actual

        # 2. TERMINE PROPORZIONALE (sempre attivo)
        # Fornisce risposta immediata proporzionale all'errore
        p_term = self.kp * error

        # 3. TERMINE INTEGRALE (solo se abilitato)
        # Accumula l'errore nel tempo per eliminare offset stazionario
        i_term = 0.0
        if self.use_integral and self.ki > 0:
            # Integrazione con metodo di Eulero: I += e·dt
            self.integral += error * self.dt

            # ANTI-WINDUP: limita accumulo per evitare saturazione eccessiva
            self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)

            i_term = self.ki * self.integral

        # 4. TERMINE DERIVATIVO (solo se abilitato)
        # Anticipa la variazione dell'errore, smorza oscillazioni
        d_term = 0.0
        if self.use_derivative and self.kd > 0:
            # Derivata con differenze finite: dE/dt ≈ (e_k - e_{k-1})/dt
            derivative = (error - self.prev_error) / self.dt
            d_term = self.kd * derivative

        # 5. COMANDO TOTALE
        delta = p_term + i_term + d_term

        # 6. SATURAZIONE ai limiti fisici dello sterzo
        delta_saturated = np.clip(delta, -self.max_delta, self.max_delta)

        # 7. ANTI-WINDUP CONDIZIONALE
        # Se siamo in saturazione, "scarica" l'integrale per evitare accumulo
        if self.use_integral and abs(delta) >= self.max_delta:
            self.integral -= error * self.dt

        # 8. AGGIORNA STATO per prossima iterazione
        self.prev_error = error

        return delta_saturated

    def reset(self):
        """
        Reset dello stato interno del controllore.
        Da chiamare all'inizio di una nuova simulazione.
        """
        self.integral = 0.0
        self.prev_error = 0.0

    def get_control_type(self) -> str:
        """
        Restituisce il tipo di controllore attivo.

        Returns:
            "P", "PI", "PD", o "PID"
        """
        if not self.use_integral and not self.use_derivative:
            return "P"
        elif self.use_integral and not self.use_derivative:
            return "PI"
        elif not self.use_integral and self.use_derivative:
            return "PD"
        else:
            return "PID"

    def get_terms(self, omega_des: float, omega_actual: float) -> dict:
        """
        Calcola e restituisce i singoli termini del PID.
        Utile per debugging e analisi.

        Returns:
            dict con 'P', 'I', 'D', 'total', 'error'
        """
        error = omega_des - omega_actual

        p_term = self.kp * error

        i_term = 0.0
        if self.use_integral:
            i_term = self.ki * self.integral

        d_term = 0.0
        if self.use_derivative:
            derivative = (error - self.prev_error) / self.dt
            d_term = self.kd * derivative

        return {
            'error': error,
            'P': p_term,
            'I': i_term,
            'D': d_term,
            'total': p_term + i_term + d_term
        }


class VelocityPIDController:
    """
    Controllore PI per velocità longitudinale.

    Controlla il duty cycle del motore per raggiungere
    la velocità desiderata vx_des.
    """

    def __init__(self,
                 kp: float = 10.0,
                 ki: float = 5.0,
                 dt: float = 0.001,
                 max_duty: float = 1.0):
        """
        Inizializza il controllore PI per velocità.

        Args:
            kp: guadagno proporzionale
            ki: guadagno integrale
            dt: passo di campionamento [s]
            max_duty: limite duty cycle [0, 1]
        """
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.max_duty = max_duty

        # Stato interno
        self.integral = 0.0
        self.integral_max = 1.0  # Anti-windup

    def compute(self, vx_des: float, vx_actual: float) -> float:
        """
        Calcola duty cycle per raggiungere vx_des.

        Args:
            vx_des: velocità desiderata [m/s]
            vx_actual: velocità misurata [m/s]

        Returns:
            d: duty cycle [0, 1]
        """
        # Errore
        error = vx_des - vx_actual

        # Termine P
        p_term = self.kp * error

        # Termine I
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.integral

        # Comando PI
        d_cmd = p_term + i_term

        # Saturazione
        d_cmd_saturated = np.clip(d_cmd, 0.0, self.max_duty)

        # Anti-windup
        if d_cmd >= self.max_duty or d_cmd <= 0.0:
            self.integral -= error * self.dt

        return d_cmd_saturated

    def reset(self):
        """Reset stato interno"""
        self.integral = 0.0