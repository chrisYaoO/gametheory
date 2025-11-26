import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt


"""data structures"""


@dataclass
class GameParams:
    N: int
    a: np.ndarray  # shape (N,)
    b: np.ndarray  # shape (N,)
    w: np.ndarray  # shape (N,)
    gamma: np.ndarray  # shape (N,)
    tau: float
    C: float
    lam: float
    eps: float


@dataclass
class RunResult:
    algo: str  # "BR" or "GP"
    N: int
    tau: float
    seed: int
    iters: int
    converged: bool
    wall_time: float
    stationarity: float
    potential: float
    x: np.ndarray  # final allocation
    fairness: Dict[str, float]

    stat_hist: np.ndarray  # stationarity per iteration
    time_hist: np.ndarray  # cumulative wall-clock time per iteration
    fairness_hist: Dict[str, np.ndarray]


"""Core model utilities"""


def positive_part(z: float) -> float:
    return max(z, 0.0)


def potential_phi(x: np.ndarray, params: GameParams) -> float:
    """
    potential function: type: float
    """
    N = params.N
    a, b, w, gamma = params.a, params.b, params.w, params.gamma
    tau, C, lam, eps = params.tau, params.C, params.lam, params.eps

    x_clipped = np.maximum(x, eps)
    X = np.sum(x_clipped)

    # self-side terms
    term_scaling = 0.5 * np.sum(a * x_clipped**2)
    term_price = np.sum(b * x_clipped)
    term_time = np.sum(gamma * w / x_clipped)

    # congestion terms
    term_cong1 = 0.5 * tau * X**2
    term_cong2 = 0.5 * tau * np.sum(x_clipped**2)

    # capacity penalty
    overload = positive_part(X - C)
    term_cap = 0.5 * lam * overload**2

    return term_scaling + term_price + term_time + term_cong1 + term_cong2 + term_cap


def grad_phi(x: np.ndarray, params: GameParams) -> np.ndarray:
    """
    gradient of phi  type: np.ndarray
    """
    a, b, w, gamma = params.a, params.b, params.w, params.gamma
    tau, C, lam, eps = params.tau, params.C, params.lam, params.eps

    x_clipped = np.maximum(x, eps)
    X = np.sum(x_clipped)
    overload = positive_part(X - C)

    g = (
        a * x_clipped
        + b
        - gamma * w / (x_clipped**2)
        + tau * (X + x_clipped)
        + lam * overload
    )
    return g


def stationarity_measure(x: np.ndarray, params: GameParams) -> float:
    g = grad_phi(x, params)
    return np.max(np.abs(g))


# Fairness metrics
def gini_coefficient(v: np.ndarray) -> float:
    v = np.asarray(v)
    if np.allclose(v, 0):
        return 0.0
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    # Gini = (2 * sum(i * v_i) / (n * sum v)) - (n + 1) / n
    numerator = 2.0 * np.sum((np.arange(1, n + 1)) * v)
    denominator = n * np.sum(v)
    return numerator / denominator - (n + 1) / n


def fairness_metrics(x: np.ndarray, params: GameParams) -> Dict[str, float]:
    x = np.asarray(x)
    urgency = params.w * params.gamma

    var_x = float(np.var(x))
    gini_x = float(gini_coefficient(x))

    if np.allclose(x, x[0]) or np.allclose(urgency, urgency[0]):
        corr = 0.0
    else:
        corr = float(np.corrcoef(x, urgency)[0, 1])

    return {
        "variance_x": var_x,
        "gini_x": gini_x,
        "corr_x_urgency": corr,
    }


# ============================
# Instance generation
# ============================


def generate_raw_params(
    N: int, tau: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a_i, b_i, w_i, gamma_i
    """
    a = rng.uniform(0.8, 1.6, size=N)
    b = np.full(N, 0.5)
    w = rng.lognormal(mean=0.0, sigma=0.6, size=N)
    gamma = rng.choice(np.array([0.5, 1.0, 2.0]), size=N)
    return a, b, w, gamma


def solve_ne_lambda0(
    N: int,
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    gamma: np.ndarray,
    tau: float,
    eps: float,
    seed: int,
    max_iter: int = 2000,
    eta: float = 5e-3,
    stat_eps: float = 1e-8,
) -> np.ndarray:
    """
    Solve the game once with λ=0 via GP, to get X_free.
    C = 0.95 * X_free
    """
    rng = np.random.default_rng(seed)
    x0 = eps * (1.0 + 1e-3 * rng.normal(size=N))

    params_lam0 = GameParams(
        N=N,
        a=a,
        b=b,
        w=w,
        gamma=gamma,
        tau=tau,
        C=1.0,  # dummy
        lam=0.0,
        eps=eps,
    )

    x = x0.copy()
    for _ in range(max_iter):
        g = grad_phi(x, params_lam0)
        x_new = np.maximum(x - eta * g, eps)
        x = x_new
        stat = stationarity_measure(x, params_lam0)
        if stat <= stat_eps:
            break

    return x


def build_game_params_for_experiment(
    N: int,
    tau: float,
    lam: float,
    eps: float,
    seed_for_instance: int,
) -> GameParams:
    """
    Generate an instance and compute C = 0.95 X_free.
    """
    rng = np.random.default_rng(seed_for_instance)
    a, b, w, gamma = generate_raw_params(N, tau, rng)

    # solve once with λ = 0 to get X_free
    x_free = solve_ne_lambda0(
        N=N,
        a=a,
        b=b,
        w=w,
        gamma=gamma,
        tau=tau,
        eps=eps,
        seed=seed_for_instance + 12345,
    )
    X_free = float(np.sum(x_free))
    C = 0.95 * X_free

    params = GameParams(
        N=N,
        a=a,
        b=b,
        w=w,
        gamma=gamma,
        tau=tau,
        C=C,
        lam=lam,
        eps=eps,
    )
    return params


# Best Response (BR) utilities
def br_positive_root_newton_bisect(
    i: int,
    x: np.ndarray,
    params: GameParams,
    max_newton: int = 5,
    newton_tol: float = 1e-10,
) -> float:
    """
    Solve the 1D cubic for player i's best response given x^k.
    We treat S_-i = X^k - x_i^k, and (X^k - C)_+ as constants,
    yielding a cubic in x_i >= eps:

    (a_i + 2τ) x^3 + (b_i + τ S_-i + λ (X^k - C)_+) x^2 - γ_i w_i = 0

    We find its unique positive root via Newton + bisection fallback.
    """
    a, b, w, gamma = params.a, params.b, params.w, params.gamma
    tau, C, lam, eps = params.tau, params.C, params.lam, params.eps

    x_clipped = np.maximum(x, eps)
    X = float(np.sum(x_clipped))
    S_minus = X - x_clipped[i]
    overload_const = positive_part(X - C)

    A = a[i] + 2.0 * tau
    B = b[i] + tau * S_minus + lam * overload_const
    C_const = -gamma[i] * w[i]

    def f(z: float) -> float:
        return A * z**3 + B * z**2 + C_const

    def df(z: float) -> float:
        return 3.0 * A * z**2 + 2.0 * B * z

    # Newton starting point: current x_i
    z = max(x_clipped[i], eps * 10.0)
    for _ in range(max_newton):
        val = f(z)
        deriv = df(z)
        if abs(deriv) < 1e-14:
            break
        step = val / deriv
        z_new = z - step
        if z_new <= eps / 2:
            # step went negative; bail to bisection later
            break
        if abs(step) < newton_tol * max(1.0, abs(z)):
            if z_new >= eps:
                return float(z_new)
            else:
                return float(eps)
        z = z_new

    # If Newton didn't converge reliably, do bisection
    # f(eps) < 0 (approx -γ w_i), and for large z, f(z) > 0
    lo = eps
    hi = max(z, 1.0)
    # Increase hi until f(hi) >= 0
    fh = f(hi)
    iter_safety = 0
    while fh < 0 and iter_safety < 60:
        hi *= 2.0
        fh = f(hi)
        iter_safety += 1

    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12 * max(1.0, mid):
            break

    root = 0.5 * (lo + hi)
    return float(max(root, eps))


def run_br(
    params: GameParams,
    seed: int,
    gamma_relax: float = 0.5,
    max_iter: int = 2000,
    stat_eps: float = 1e-6,
) -> RunResult:
    N = params.N
    rng = np.random.default_rng(seed)
    x = params.eps * (1.0 + 1e-3 * rng.normal(size=N))

    # histories
    stat_hist = []
    time_hist = []
    fairness_hist = {
        "variance_x": [],
        "gini_x": [],
        "corr_x_urgency": [],
    }

    t0 = time.perf_counter()
    converged = False

    for k in range(max_iter):
        x_old = x.copy()
        # Jacobi BR
        br_vals = np.zeros_like(x_old)
        for i in range(N):
            br_i = br_positive_root_newton_bisect(i, x_old, params)
            br_vals[i] = br_i
        x = x_old + gamma_relax * (br_vals - x_old)
        x = np.maximum(x, params.eps)

        # measure stationarity + fairness
        stat = stationarity_measure(x, params)
        fair = fairness_metrics(x, params)

        # record histories
        stat_hist.append(stat)
        time_hist.append(time.perf_counter() - t0)
        fairness_hist["variance_x"].append(fair["variance_x"])
        fairness_hist["gini_x"].append(fair["gini_x"])
        fairness_hist["corr_x_urgency"].append(fair["corr_x_urgency"])

        if stat <= stat_eps:
            converged = True
            iters = k + 1
            break
    else:
        iters = max_iter
        stat = stationarity_measure(x, params)
        fair = fairness_metrics(x, params)

    t1 = time.perf_counter()
    phi_val = potential_phi(x, params)
    fairness_final = fairness_metrics(x, params)

    stat_hist = np.asarray(stat_hist)
    time_hist = np.asarray(time_hist)
    fairness_hist = {k: np.asarray(v) for k, v in fairness_hist.items()}

    return RunResult(
        algo="BR",
        N=params.N,
        tau=params.tau,
        seed=seed,
        iters=iters,
        converged=converged,
        wall_time=t1 - t0,
        stationarity=float(stat),
        potential=float(phi_val),
        x=x,
        fairness=fairness_final,
        stat_hist=stat_hist,
        time_hist=time_hist,
        fairness_hist=fairness_hist,
    )


# ============================
# Gradient Play (GP)
# ============================


def run_gp(
    params: GameParams,
    seed: int,
    eta: float = 5e-3,
    max_iter: int = 2000,
    stat_eps: float = 1e-6,
    max_step: float = 1,
) -> RunResult:
    N = params.N
    rng = np.random.default_rng(seed)

    x = np.maximum(rng.lognormal(mean=0.0, sigma=0.3, size=N), params.eps)

    # histories
    stat_hist = []
    time_hist = []
    fairness_hist = {
        "variance_x": [],
        "gini_x": [],
        "corr_x_urgency": [],
    }

    t0 = time.perf_counter()
    converged = False

    for k in range(max_iter):
        g = grad_phi(x, params)

        step = -eta * g
        step = np.clip(step, -max_step, max_step)
        x = np.maximum(x + step, params.eps)

        stat = stationarity_measure(x, params)
        fair = fairness_metrics(x, params)

        stat_hist.append(stat)
        time_hist.append(time.perf_counter() - t0)
        fairness_hist["variance_x"].append(fair["variance_x"])
        fairness_hist["gini_x"].append(fair["gini_x"])
        fairness_hist["corr_x_urgency"].append(fair["corr_x_urgency"])

        if stat <= stat_eps:
            converged = True
            iters = k + 1
            break
    else:
        iters = max_iter
        stat = stationarity_measure(x, params)
        fair = fairness_metrics(x, params)

    t1 = time.perf_counter()
    phi_val = potential_phi(x, params)
    fairness_final = fairness_metrics(x, params)

    stat_hist = np.asarray(stat_hist)
    time_hist = np.asarray(time_hist)
    fairness_hist = {k: np.asarray(v) for k, v in fairness_hist.items()}

    return RunResult(
        algo="GP",
        N=params.N,
        tau=params.tau,
        seed=seed,
        iters=iters,
        converged=converged,
        wall_time=t1 - t0,
        stationarity=float(stat),
        potential=float(phi_val),
        x=x,
        fairness=fairness_final,
        stat_hist=stat_hist,
        time_hist=time_hist,
        fairness_hist=fairness_hist,
    )


def plot_curves_for_instance(br_result: RunResult, gp_result: RunResult, prefix: str):
    """
    For fixed (N, tau, seed), plot:
      - stationarity vs iteration
      - stationarity vs wall-clock time
      - fairness (Gini) vs iteration
    Each plot: BR & GP on the same figure.
    Files saved as: f"{prefix}_stat_vs_iter.png", etc.
    """
    # --- 1) stationarity vs iteration ---
    plt.figure()
    plt.plot(br_result.stat_hist, label="BR")
    plt.plot(gp_result.stat_hist, label="GP")
    plt.xlabel("Iteration")
    plt.ylabel("Stationarity")
    plt.title(
        f"Stationarity vs Iter (N={br_result.N}, tau={br_result.tau}, seed={br_result.seed})"
    )
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_stat_vs_iter.png", dpi=200)
    plt.close()

    # --- 2) stationarity vs real timeline ---
    plt.figure()
    plt.plot(br_result.time_hist, br_result.stat_hist, label="BR")
    plt.plot(gp_result.time_hist, gp_result.stat_hist, label="GP")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Stationarity")
    plt.title(
        f"Stationarity vs Time (N={br_result.N}, tau={br_result.tau}, seed={br_result.seed})"
    )
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_stat_vs_time.png", dpi=200)
    plt.close()

    # --- 3) fairness (Gini) vs iteration ---
    plt.figure()
    br_gini = br_result.fairness_hist["gini_x"]
    gp_gini = gp_result.fairness_hist["gini_x"]
    plt.plot(br_gini, label="BR")
    plt.plot(gp_gini, label="GP")
    plt.xlabel("Iteration")
    plt.ylabel("Gini coefficient of x")
    plt.title(
        f"Fairness (Gini) vs Iter (N={br_result.N}, tau={br_result.tau}, seed={br_result.seed})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_fairness_vs_iter.png", dpi=200)
    plt.close()


# ============================
# Experiment harness
# ============================


def run_experiments():
    Ns = [20, 100]
    taus = [0.1, 0.3, 0.6]
    lam = 10.0
    eps = 1e-6
    seeds = [0, 1, 2, 3, 5]

    gp_eta = 5e-3
    stat_eps = 1e-6
    max_iter = 2000

    all_results: List[RunResult] = []

    for N in Ns:
        for tau in taus:
            print(f"\n=== N={N}, tau={tau} ===")
            for seed in seeds:
                params = build_game_params_for_experiment(
                    N=N,
                    tau=tau,
                    lam=lam,
                    eps=eps,
                    seed_for_instance=1000 * (N + 1) + 10 * int(tau * 10) + seed,
                )

                # ----- BR -----
                br_result = run_br(
                    params=params,
                    seed=seed,
                    gamma_relax=0.5,
                    max_iter=max_iter,
                    stat_eps=stat_eps,
                )
                all_results.append(br_result)
                print(
                    f"BR  seed={seed:2d}  iters={br_result.iters:4d}  "
                    f"conv={br_result.converged}  time={br_result.wall_time:.4f}  "
                    f"stat={br_result.stationarity:.2e}"
                )

                # ----- GP -----
                gp_result = run_gp(
                    params=params,
                    seed=seed,
                    eta=gp_eta,
                    max_iter=max_iter,
                    stat_eps=stat_eps,
                )
                all_results.append(gp_result)
                print(
                    f"GP  seed={seed:2d}  iters={gp_result.iters:4d}  "
                    f"conv={gp_result.converged}  time={gp_result.wall_time:.4f}  "
                    f"stat={gp_result.stationarity:.2e}"
                )

                if seed == 1:
                    prefix = f"results/N{N}_tau{tau}_seed{seed}"
                    plot_curves_for_instance(br_result, gp_result, prefix)

    # Summary
    print("\n================ Summary (mean over seeds) ================")
    for N in Ns:
        for tau in taus:
            for algo in ["BR", "GP"]:
                sub = [
                    r
                    for r in all_results
                    if r.N == N and r.tau == tau and r.algo == algo
                ]
                if not sub:
                    continue
                iters_mean = np.mean([r.iters for r in sub])
                time_mean = np.mean([r.wall_time for r in sub])
                stat_mean = np.mean([r.stationarity for r in sub])
                pot_mean = np.mean([r.potential for r in sub])
                conv_rate = np.mean([1.0 if r.converged else 0.0 for r in sub])

                # 平均 fairness
                var_mean = np.mean([r.fairness["variance_x"] for r in sub])
                gini_mean = np.mean([r.fairness["gini_x"] for r in sub])
                corr_mean = np.mean([r.fairness["corr_x_urgency"] for r in sub])

                print(
                    f"Algo={algo}, N={N:3d}, tau={tau:.1f}: "
                    # f"conv_rate={conv_rate:.2f}, "
                    f"iters_mean={iters_mean:.1f}, "
                    f"time_mean={time_mean:.4f}, "
                    f"stat_mean={stat_mean:.2e}, "
                    f"phi_mean={pot_mean:.4e}, "
                    # f"var_mean={var_mean:.4e}, "
                    f"gini_mean={gini_mean:.4f}, "
                    # f"corr_mean={corr_mean:.4f}"
                )

    return all_results


if __name__ == "__main__":
    run_experiments()
