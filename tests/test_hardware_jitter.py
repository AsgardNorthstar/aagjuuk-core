import time
import numpy as np
from aagjuuk_control_loop import ThermalStressController

def test_control_loop_under_latency_jitter():
    """
    Stress-tests the controller PID calculation step against simulated
    hardware communications delay (jitter) to ensure mechanical stability.
    """
    controller = ThermalStressController(target_stress_limit=25.0)
    current_power = 50.0
    
    # Simulate a high-speed telemetry sequence with randomized latency spikes
    for _ in range(50):
        # Generate physical stress telemetry with Gaussian white noise
        simulated_stress = 22.0 + np.random.normal(0, 0.5)
        
        # Introduce a controlled delay representing industrial network jitter
        jitter_delay = np.random.uniform(0.001, 0.005) # 1ms to 5ms delay
        time.sleep(jitter_delay)
        
        start_time = time.perf_counter()
        next_power, error = controller.compute_control_action(
            current_stress_max=simulated_stress,
            current_laser_power=current_power
        )
        latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Assertions to guarantee runtime constraints hold true even under jitter
        assert latency < 2.0  # Core math execution must remain well under 2ms limit
        assert 10.0 <= next_power <= 100.0  # Crucial: Assure safety boundaries are never breached
        current_power = next_power
