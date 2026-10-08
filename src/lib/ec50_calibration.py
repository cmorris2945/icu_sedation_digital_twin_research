from scipy.optimize import minimize_scalar


def calibrate_ec50(sas_times_min, sas_values, n_calib, ce_traj, params):
    """Fit EC50 to usable calibration points or use the population value."""
    usable_indices = [
        i for i in range(n_calib)
        if 0 <= sas_times_min[i] < len(ce_traj)
    ]
    if not usable_indices:
        return params['EC50_pop']

    def loss(ec50_candidate):
        total = 0.0
        for i in usable_indices:
            prediction = params['E0'] - params['Emax'] * (
                ce_traj[sas_times_min[i]] ** params['gamma']
            ) / (
                ec50_candidate ** params['gamma']
                + ce_traj[sas_times_min[i]] ** params['gamma']
            )
            total += (prediction - sas_values[i]) ** 2
        return total / len(usable_indices)

    try:
        result = minimize_scalar(loss, bounds=(0.5, 10.0), method='bounded')
        return result.x
    except Exception:
        return params['EC50_pop']