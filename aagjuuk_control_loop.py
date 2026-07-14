import numpy as np
import logging

# Set up clean hardware logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AagjuukControl")

class ThermalStressController:
    """
    Active PID hardware controller mapped to estimated physical strain fields.
    Optimized for sub-5ms edge loop frequencies.
    """
    def __init__(self, target_stress_limit: float = 25.0, kp: float = 0.4, ki: float = 0.1, kd: float = 0.05):
        self.target_stress = target_stress_limit
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral_error = 0.0
        self.last_error = 0.0
        
        # Anti-windup threshold to prevent integral runaway during rapid thermal cycling
        self.windup_guard = 10.0 

    def compute_control_action(self, current_stress_max: float, current_laser_power: float) -> tuple[float, float]:
        """
        Calculates next-step laser power percentage using anti-windup PID mechanics.
        """
        # Defensive check: Protect hardware against sensor dropouts / NaN values
        if np.isnan(current_stress_max) or np.isinf(current_stress_max):
            logger.error("Inference stress is NaN/Inf! Emergency backup power scaling triggered.")
            return min(current_laser_power * 0.5, 15.0), 0.0 # Force low-power safety state

        error = self.target_stress - current_stress_max
        
        # Proportional-Integral-Derivative terms
        p_term = self.kp * error
        
        # Anti-windup integration step
        if abs(error) < self.windup_guard:
            self.integral_error += error
        i_term = self.ki * self.integral_error
        
        d_term = self.kd * (error - self.last_error)
        self.last_error = error
        
        adjustment = p_term + i_term + d_term
        new_power = current_laser_power + adjustment
        
        # Hard safety constraints (Clamping to physical tool limits)
        # TODO: Read hardware envelope limits dynamically from active profile instead of hardcoding [10, 100]
        clamped_power = float(np.clip(new_power, 10.0, 100.0))
        
        return clamped_power, error
