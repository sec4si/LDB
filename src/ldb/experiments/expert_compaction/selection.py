"""Frozen local response costs; no downstream validation labels enter selection."""

import numpy as np


def response_costs(outputs, routes):
    """outputs: expert x token x feature; routes: token x expert, post top-k."""
    outputs = np.asarray(outputs, dtype=np.float64)
    routes = np.asarray(routes, dtype=np.float64)
    if outputs.ndim != 3 or routes.shape != outputs.shape[1::-1]:
        raise ValueError("Incompatible response and routing dimensions")
    if not np.isfinite(outputs).all() or not np.isfinite(routes).all():
        raise ValueError("Nonfinite fingerprint inputs")
    n = len(outputs)
    weighted = np.empty((n, n))
    distance = np.empty((n, n))
    for target in range(n):
        error = np.square(outputs - outputs[target]).sum(axis=-1)
        distance[target] = error.mean(axis=-1)
        weighted[target] = (error * routes[:, target] ** 2).mean(axis=-1)
    return weighted, distance


def choose_map(cost, counts, remove, method, seed=0):
    """Greedy medoid deletion or frequency/random controls, always without chains."""
    cost = np.asarray(cost)
    n = len(counts)
    if cost.shape != (n, n) or not 0 <= remove < n or not np.isfinite(cost).all():
        raise ValueError("Invalid cost matrix or removal count")
    retained = list(range(n))
    if method == "fingerprint":
        for _ in range(remove):
            # Charge all original router slots, including earlier merged slots.
            losses = [
                cost[:, [e for e in retained if e != candidate]].min(axis=1).sum()
                for candidate in retained
            ]
            retained.pop(int(np.argmin(losses)))
    elif method in {"frequency", "random_donor"}:
        deleted = set(np.argsort(counts, kind="stable")[:remove].tolist())
        retained = [e for e in retained if e not in deleted]
    elif method == "random":
        retained = sorted(np.random.default_rng(seed).choice(n, n - remove, replace=False).tolist())
    else:
        raise ValueError(f"Unknown selection method: {method}")
    mapping = np.array(retained)[cost[:, retained].argmin(axis=1)]
    if method in {"random", "random_donor"}:
        rng = np.random.default_rng(seed)
        mapping = rng.choice(retained, n)
    mapping[retained] = retained
    return mapping.tolist()


def allocate_layers(costs, budgets, total):
    """Exact finite-budget allocation for an additive layer-cost surrogate."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != len(budgets) or not np.isfinite(costs).all():
        raise ValueError("Invalid layer cost table")
    states = {0: (0.0, [])}
    for row in costs:
        updated = {}
        for used, (loss, choices) in states.items():
            for count, value in zip(budgets, row):
                after = used + count
                if after <= total and (after not in updated or loss + value < updated[after][0]):
                    updated[after] = (loss + value, choices + [count])
        states = updated
    if total not in states:
        raise ValueError("Requested removal budget is infeasible")
    return states[total]
