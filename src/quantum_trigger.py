import torch
import torch.nn as nn
import pennylane as qml

class LowPowerQuantumTrigger(nn.Module):
    def __init__(self):
        super(LowPowerQuantumTrigger, self).__init__()
        # Initialize a single virtual qubit simulator
        self.dev = qml.device("default.qubit", wires=1)
        
        # Build the QNode using PyTorch's backend for calculations
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        
        # Trainable internal parameter to fine-tune sensitivity
        self.bias = nn.Parameter(torch.tensor([0.2]))

    def _circuit(self, single_feature, bias_param):
        # Rotate the qubit based on the physical signal input
        qml.RX(single_feature, wires=0)
        # Apply internal baseline tuning adjustments
        qml.RY(bias_param[0], wires=0)
        # Measure spin orientation along the Z-axis (returns a value between -1 and 1)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_expectation = self.qnode(x, self.bias)
        # Translate the [-1, 1] output cleanly into a [0, 1] confidence probability
        confidence = (1.0 - raw_expectation) / 2.0
        return confidence
