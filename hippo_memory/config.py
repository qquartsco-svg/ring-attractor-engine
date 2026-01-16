"""
Hippocampus Memory Engine - Configuration

Hodgkin-Huxley neuron model configuration
"""

# HH Soma Configuration
HH_CONFIG = {
    "V0": -70.0,
    "gNa": 220.0,
    "gK": 26.0,
    "gL": 0.02,
    "ENa": 50.0,
    "EK": -77.0,
    "EL": -54.4,
    "spike_thresh": -15.0,
    "use_pump": True,
    "g_pump": 0.2,
    "E_pump": -70.0,
    "ATP0_ref": 100.0,
    "g_pump_consume": 0.02,
}

# For compatibility with existing code
CONFIG = {
    "HH": HH_CONFIG.copy()
}
