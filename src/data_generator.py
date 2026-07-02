import torch
import numpy as np

def generate_ambient_stream(num_seconds: int = 20, sample_rate_hz: int = 100):
    """
    Simulates continuous real-world microphone data.
    Returns a sequence of raw energy readings from the sensor.
    """
    np.random.seed(42)
    total_samples = num_seconds * sample_rate_hz
    
    # 1. Generate normal ambient background noise (low fluctuations)
    stream = np.random.normal(loc=0.1, scale=0.05, size=total_samples)
    
    # 2. Inject intentional "Wake Word" spikes at specific times
    # We add voice spikes at Second 5, Second 12, and Second 17
    spike_locations = [5 * sample_rate_hz, 12 * sample_rate_hz, 17 * sample_rate_hz]
    
    for loc in spike_locations:
        # Create a gradual ramp up and down representing someone saying a word
        window = np.sin(np.linspace(0, np.pi, 20)) * 1.5
        stream[loc:loc+20] += window
        
    # Keep data strictly bounded between 0 and 2
    stream = np.clip(stream, 0.0, 2.0)
    
    # CRITICAL STEP: Scale data directly to quantum rotation constraints [0, pi]
    quantum_ready_stream = (stream / 2.0) * np.pi
    
    return torch.tensor(quantum_ready_stream, dtype=torch.float32)

if __name__ == "__main__":
    stream = generate_ambient_stream(2)
    print(f"[SUCCESS] Generated mock sensor stream of length {len(stream)} samples.")
