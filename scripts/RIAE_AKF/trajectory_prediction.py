import CoordsKF
import numpy as np


measurements = [
    [0.00, 0.0, 0.0, 1.0],
    [0.02, 0.02, -0.01, 0.98],
    [0.04, 0.04, -0.02, 0.94],
    [0.06, 0.06, -0.03, 0.90],
]

# Initialize KF
kf = CoordsKF(
    dt0=0.02,
    initial_position=[0, 0, 1],
    initial_velocity=[1, 0, 0]
)

# Run RIAE-AKF
state_pred = kf.RIAE_AKF(
    measurements,
    future_dt=0.02,
    adapt_Q=True,
    alpha=0.05,
    window_size=3,
    conf_level=0.99,
    warmup_steps=2,
    robust_update=True
)

print("Future predicted state:", state_pred)
