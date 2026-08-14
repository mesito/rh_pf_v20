"""
Shared configuration for RH partial-fraction numerical verifications.
Paper: "Off-line zeros of the Riemann xi-function: a constraint network, an
        exactly solvable collision model, an energy-budget divergence, and a
        lifetime-deficit dictionary with negative controls  (version 18.4)"
Author: Mesut Ismail, PhD Student, TU-Sofia
Repository: https://github.com/mesito/rh_pf_v20

All computations use mpmath at 25-digit precision.
Zeros are loaded in batches of 200 (10 passes for 2000 total).
"""

import mpmath
import os
import pickle
import time

# Precision
mpmath.mp.dps = 25

# Constants
A_CONST = 0.301        # auxiliary constant for V'
BATCH_SIZE = 200       # zeros per batch
N_BATCHES = 10         # total batches
TOTAL_ZEROS = BATCH_SIZE * N_BATCHES  # 2000
N_COEFF = 60000  # coefficient sieve depth for the supply envelope

# Cache file
CACHE_FILE = os.path.join(os.path.dirname(__file__), "zeros_cache.pkl")


def load_zeros_batch(start, end):
    """Load zeta zeros from index start to end (1-indexed)."""
    zeros = []
    for n in range(start, end + 1):
        g = mpmath.im(mpmath.zetazero(n))
        zeros.append(g)
    return zeros


def load_all_zeros(verbose=True):
    """
    Load all 2000 zeros in 10 batches of 200.
    Caches to disk for reuse.
    """
    if os.path.exists(CACHE_FILE):
        if verbose:
            print(f"Loading cached zeros from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            zeros = pickle.load(f)
        if len(zeros) >= TOTAL_ZEROS:
            if verbose:
                print(f"  Loaded {len(zeros)} zeros (gamma_max ~ {float(zeros[-1]):.1f})")
            return zeros[:TOTAL_ZEROS]

    zeros = []
    for batch in range(N_BATCHES):
        start = batch * BATCH_SIZE + 1
        end = (batch + 1) * BATCH_SIZE
        t0 = time.time()
        if verbose:
            print(f"  Batch {batch+1}/{N_BATCHES}: zeros {start}-{end} ...", end="", flush=True)
        batch_zeros = load_zeros_batch(start, end)
        zeros.extend(batch_zeros)
        dt = time.time() - t0
        if verbose:
            print(f" done ({dt:.1f}s, gamma_max ~ {float(batch_zeros[-1]):.1f})")

    # Cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(zeros, f)
    if verbose:
        print(f"  Cached {len(zeros)} zeros to {CACHE_FILE}")

    return zeros


def Son(t0, zeros):
    """On-line zero sum: Sum_k 1/(t0 - gamma_k)^2"""
    s = mpmath.mpf(0)
    for gk in zeros:
        diff = t0 - gk
        if abs(diff) > 1e-15:
            s += 1 / diff**2
    return s


def gap_midpoint(zeros, gap_idx):
    """Return midpoint t0 of gap between zeros[gap_idx] and zeros[gap_idx+1]."""
    return (zeros[gap_idx] + zeros[gap_idx + 1]) / 2


def print_header(title):
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
