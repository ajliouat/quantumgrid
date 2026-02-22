"""Build PennyLane-compatible cost Hamiltonians from QUBO / Ising models.

Provides helper functions that construct PennyLane ``Hamiltonian`` objects
from the Ising coefficients produced by ``formulation.qubo_encoding``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pennylane as qml


def ising_to_pennylane(
    J: np.ndarray,
    h: np.ndarray,
    offset: float = 0.0,
) -> qml.Hamiltonian:
    """Build a PennyLane Hamiltonian from Ising coefficients.

    H = sum_{i<j} J_{ij} Z_i Z_j  +  sum_i h_i Z_i  +  offset * I

    Args:
        J: (n, n) upper-triangular coupling matrix.
        h: (n,) local field vector.
        offset: constant offset (energy shift).

    Returns:
        PennyLane Hamiltonian.
    """
    n = len(h)
    coeffs: List[float] = []
    obs: List[qml.operation.Observable] = []

    # Constant offset: offset * I
    if abs(offset) > 1e-12:
        coeffs.append(offset)
        obs.append(qml.Identity(0))

    # Linear terms: h_i * Z_i
    for i in range(n):
        if abs(h[i]) > 1e-12:
            coeffs.append(float(h[i]))
            obs.append(qml.PauliZ(i))

    # Quadratic terms: J_{ij} * Z_i Z_j
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                coeffs.append(float(J[i, j]))
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    if not coeffs:
        coeffs.append(0.0)
        obs.append(qml.Identity(0))

    return qml.Hamiltonian(coeffs, obs)


def mixer_hamiltonian(n_qubits: int) -> qml.Hamiltonian:
    """Standard X-mixer Hamiltonian for QAOA.

    H_mixer = sum_i X_i

    Args:
        n_qubits: Number of qubits.

    Returns:
        PennyLane Hamiltonian.
    """
    coeffs = [1.0] * n_qubits
    obs = [qml.PauliX(i) for i in range(n_qubits)]
    return qml.Hamiltonian(coeffs, obs)


def expectation_value(
    hamiltonian: qml.Hamiltonian,
    state: np.ndarray,
) -> float:
    """Compute exact expectation <state|H|state> via matrix.

    Args:
        hamiltonian: PennyLane Hamiltonian.
        state: (2^n,) complex statevector.

    Returns:
        Real expectation value.
    """
    mat = qml.matrix(hamiltonian)
    return float(np.real(state.conj() @ mat @ state))


def bitstring_energy(
    hamiltonian: qml.Hamiltonian,
    bitstring: np.ndarray,
) -> float:
    """Compute energy of a computational-basis bitstring.

    Args:
        hamiltonian: PennyLane Hamiltonian.
        bitstring: (n,) array of {0, 1}.

    Returns:
        Energy E = <z|H|z> where z = (-1)^bitstring (Ising convention).
    """
    spins = 1 - 2 * bitstring.astype(float)  # 0 -> +1, 1 -> -1
    mat = qml.matrix(hamiltonian)
    n = len(bitstring)
    # Build basis state index
    idx = int(sum(int(b) << (n - 1 - i) for i, b in enumerate(bitstring)))
    return float(np.real(mat[idx, idx]))


def num_hamiltonian_terms(hamiltonian: qml.Hamiltonian) -> int:
    """Return number of Pauli terms in a Hamiltonian."""
    return len(hamiltonian.coeffs)
