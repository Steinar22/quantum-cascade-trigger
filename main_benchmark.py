import torch
import time
from src.data_generator import generate_ambient_stream
from src.quantum_trigger import LowPowerQuantumTrigger

# --- SIMULATION PARAMETERS ---
TRIGGER_THRESHOLD = 0.65  # Confidence required to wake up the main AI
TOTAL_SECONDS = 20
HZ = 100

# 1. Initialize the stream and the quantum filter
stream = generate_ambient_stream(num_seconds=TOTAL_SECONDS, sample_rate_hz=HZ)
quantum_filter = LowPowerQuantumTrigger()

classical_active_cycles = 0
total_cycles = len(stream)

print("============ RUNNING CASCADED SIMULATION ============")

for step, sample in enumerate(stream):
    # Run the ultra-fast 1-qubit tripwire calculation
    with torch.no_grad():
        trigger_confidence = quantum_filter(sample)
        
    # Check if the signal crosses our decision threshold
    if trigger_confidence.item() >= TRIGGER_THRESHOLD:
        # WAKE UP STAGE 2 (The heavy classical model)
        classical_active_cycles += 1
        current_second = step / HZ
        print(f"[ALERT] Time: {current_second:.2f}s | Signal: {sample.item():.2f} | Confidence: {trigger_confidence.item():.2f} -> WAKING UP HEAVY CLASSICAL AI")
    else:
        # Keep Stage 2 asleep, saving maximum battery energy
        pass

# 2. Calculate Efficiency Metrics
power_saved_percentage = (1.0 - (classical_active_cycles / total_cycles)) * 100

print("\n================ FINAL INVENTION REPORT ================")
print(f"Total Evaluated Signal Cycles  : {total_cycles}")
print(f"Cycles Classical Model Was Awake: {classical_active_cycles}")
print(f"Total Compute/Power Saved      : {power_saved_percentage:.2f}%")
print("========================================================")
