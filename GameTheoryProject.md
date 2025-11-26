# Game Theory Project

## Best Response vs Gradient Play for Congested Cloud Allocation: Convergence, Information, and Performance

Topic: Type 4 (d): continuous action games: Best-response (BR) and gradient play (GP)

### Abstract

We study a repeated N-player continuous-action game that models cloud computational resource allocation under congestion and deadline pressure. Each tenant i chooses a non-negative and continuous resource rate (GPU-hours per unit time) to minimize a cost function that combines (i) self-side scaling cost with diminishing returns, (ii) a baseline price , (iii) a congestion externality that grows with total load and (iv) a time-loss penalty inversely proportional to allocated rate, capturing the incentive to finish sooner for jobs. The game comes with a strictly convex potential, guaranteeing a unique Nash equilibrium (NE). We compare two lightweight learning dynamics— Best Response (BR), implemented as block coordinate descent via 1D cubic solves, and Gradient Play (GP), a projected gradient descent on the potential. Simulations across heterogeneous job sizes, and capacity penalties evaluate convergence speed, wall-clock time, social cost and utilization. These results clarify the algorithmic trade-off between per-iteration accuracy and decentralized implementability, and suggest tuning congestion/price signals to steer the decentralized equilibrium toward efficient, time-aware allocations in multi-tenant clouds.



### Introduction

Modern cloud platforms must allocate shared compute (e.g., CPU/GPU time, memory bandwidth) to many independent tenants who arrive with heterogeneous workloads and urgency. Centralized schedulers are powerful but often rely on detailed per-job information and tight coordination that is costly to collect in real time. A complementary approach is **decentralized adjustment**: each tenant adapts its requested rate based on minimal aggregate feedback, and the system’s operating point emerges as the equilibrium of a repeated game. This projectadopts that view and studies two canonical learning dynamics—**Best Response (BR)** and **Gradient Play (GP)**—in an infinite action  continuous kernel game that captures congestion and deadline pressure in multi-tenant clouds.

We model tenant ’s choice as a nonnegative resource rate. The private cost combines a self-side scaling term capturing diminishing returns and operational frictions, a baseline per-unit price, a **congestion externality** that grows with total load and lastly, a **time penalty** inversely proportional to the allocated rate to capture the desire to finish sooner for each tenant. The induced game is an **exact potential game** whose potential is strictly convex under mild assumptions, ensuring the existence and uniqueness of a Nash equilibrium (NE). Importantly, players do **not** require the full action profile of others: an aggregate congestion signal suffices to implement both BR and GP, which aligns with privacy and scalability constraints of real cloud deployments.

Our goal is not to design a new allocator but to **characterize, contrast, and illustrate** properties of BR and GP as learning mechanisms that drive the system toward equilibrium. We focus on:

- **Convergence properties and rates:** When the potential is strongly convex, GP behaves like projected gradient descent; BR acts as block coordinate descent that often achieves faster decrease per iteration, especially on “stiffer” instances (high congestion or strong time penalties).
- **Game classes covered:** exact potential games with continuous, convex costs,  a class that encompasses common congestion models used for resource scheduling.
- **Information requirements:** both dynamics operate with a broadcast aggregate signal.
- **Performance metrics:** iterations and wall-clock time to reach a target optimality gap, total cost, utilization, and fairness of the final allocation.

Our contributions are threefold. First, we formalize the cloud allocation model and prove that it yields a unique NE via a strictly convex potential. Second, we present implementable BR and GP updates (BR via 1D root finding; GP via projected gradient steps with lower-bound safeguards) that use only aggregate feedback. Third, through controlled experiments varying congestion, job sizes, and capacity pressure, we provide a clear head-to-head comparison: **BR** typically attains a given optimality gap in fewer iterations and is less sensitive to conditioning, while **GP** offers cheaper, communication-light iterations that scale naturally with the number of tenants. We conclude with practical guidance on when to prefer each dynamic and how congestion/price signals can steer decentralized behavior toward efficient, time-aware allocations.



### Model & Classes of Games

In a cloud computing scenario, each player(tenant) $P_i$ needs to choose a continuous and nonnagative resource rate $x_i$ to minimize its private cost. The cost considers those factors:

* Time penalty (finish-sooner inventive): $\gamma_i \frac{w_i}{x_i}$, where $\gamma_i \gt 0$ is the time value and $w_i$ is the job size.

* Private scaling cost with marginal diminishing returns: $\frac{a_i}{2}x_i^2$, with $a_i \gt0$
* Baseline price: $b_ix_i$, with $b_i \gt0$
* Congestion impact: $\tau X x_i$, where $X = \sum_j x_j$ is the overall system load and $\tau\gt0$ is the congestion rate. (Intuition: heavier load makes each unit slower and thus more expensive)
* Quadratic penalty: $p(X) = \frac{\lambda}{2}[X-C]_{+}^2$, with $\lambda \gt 0$

Therefore we can derive the cost function for each player: $J_i(x_i,X) = \gamma_i \frac{w_i}{x_i} + \frac{a_i}{2}x_i^2+ b_ix_i+ \tau X x_i$

And the objective for each player is:
$$
\min J_i(x_i,X)+p(X)\\
s.t.\quad x_i\ge\epsilon,\\
X = \sum_ix_i
$$
The soft capacity limits overload while avoiding per-step feasibility projections or dual updates and keep the game within the potential-game class:
$$
\Phi(x) = \sum_{i=1}^N(\frac{a_i}{2}x_i^2+b_ix_i+\gamma_i\frac{w_i}{x_i})+\frac{\tau}{2}X^2+
\frac{\tau}{2}\sum_{i=1}^Nx_i^2+\frac{\lambda}{2}[X-C]_{+}^2\\
\frac{\partial \Phi}{\partial x_i} = a_ix_i+b_i-\frac{\gamma_i w_i}{x_i^2}+\tau(X+x_i)+\lambda(X-C)_+
$$
Which matched the objective function for each player:
$$
\frac{\partial}{\partial x_i}(J_i(x_i,X)+p(X))=a_ix_i+b_i-\frac{\gamma_i w_i}{x_i^2}+\tau(X+x_i)+\lambda(X-C)_+
$$
Hence the game is an exact potential game: $\nabla_{x_i}J_i = \nabla_{x_i}\Phi$.Because all terms are convex on $x_i>\epsilon $ and $ a_i,\tau,\gamma_i w_i,\lambda>0$, $\Phi$ is **strictly convex**, so the NE is **unique** and coincides with the **unique global minimizer** of $\Phi$. Although BR/GP update players using their **own** objective, the equality$\nabla_{x_i}J_i = \nabla_{x_i}\Phi$ implies that **both dynamics decrease the same scalar function** $\Phi$. This gives: (i) uniqueness of the limit, (ii) monotone descent (Lyapunov), (iii) rate statements (strong convexity), and (iv) unified stopping criteria via the optimality gap $\Phi(x^t)-\Phi^\star$.

The game now belongs to repeated N player infinite action continuous kernel game. We can use best response play(BR) or Gradient Play(GR) to update $x_i$ iteratively. In realization, the player doesn't have to know other's response, as long as he knows the overall system load $X$.



### Algorithms

1. Best response Play (BR)

   At iteration $k$, let $S_{-i}^k = X^k-x_i^k$. $P_i$'s best response to the others solves:
   $$
   BR_i(x_i^k) = \arg\min_{x_i\ge\epsilon}[J_i(x_i,X)+p(X)]
   $$
   For this model, the first-order condition yields a one-dimentional cubic in $x_i \ge \epsilon$:
   $$
   \frac{\partial \Phi}{\partial x_i} = a_ix_i+b_i-\frac{\gamma_i w_i}{x_i^2}+\tau(X+x_i)+\lambda(X-C)_+ =0\\
   a_ix_i^3+b_ix_i^2-\gamma_i w_i+\tau(2x_i+S_{-i})x_i^2+\lambda x_i^2 (X-C)_+ =0\\
   (a_i+2\tau)x_i^3+(b_i+\tau S_{-i}^k+\lambda(X-C)_+)x_i^2-\gamma_i w_i =0
   $$
   Which has a unique positive root. We compute the best response via a few Newton steps and then apply a relaxed synchronous update:
   $$
   x_i^{k+1} = x_i^k+\gamma^k[BR_i(x^k_{-i})-x_i^k], \gamma^k\in(0,1]
   $$
   

   In all experiments we use a constant relaxation $\gamma^k=0.5$, which stabilizes the Jacobi-style simultaneous updates while preserving fast progress.

   We terminate the process when the stationary measure $\|\nabla \Phi(x^{k})\|_{\infty}\le \varepsilon_{stat}$. As mentioned in the potential game property, this relaxed BR converges globally to the unique NE.

2. Gradient play

   At iteration $k$, the partial gradient is:
   $$
   g_i(x_i^k,x_{-i}^k)
   = \frac{\partial}{\partial x_i}\big(J_i(x_i,X)+p(X)\big)\Big|_{x=(x_i^k,x_{-i}^k)}\\
   = a_i x_i^{k} + b_i - \frac{\gamma_i w_i}{(x_i^{k})^{2}}
     + \tau\big(X^{k} + x_i^{k}\big) + \lambda\,(X^{k}-C)_+
   $$

   We use the projected gradient play in our experiment:
   
   $$
   x_i^{k+1} = \big[x_i^k-\eta^kg_i(x_i^k,x_{-i}^k)\big]_{[\epsilon,\infty)}
   $$
   Where $[\cdot]_{[\epsilon,\infty)} = \max\{\cdot, \epsilon\}$ enforced the lower bound and $\eta^k \gt 0$ si the stepsize.

   We terminate the process as the same in BR play when $\|\nabla \Phi(x^{k})\|_{\infty}\le \varepsilon_{stat}$. As mentioned in the potential game property, choosing a sufficiently small stepsize ensures the potential function decreases monotonically and converges to its **unique minimizer**, which coincides with the unique Nash equilibrium.

   

### Experiment Methodology

1. instances and parameterization

   We generate multi-player cases with heterogeneous costs and job sizes:

   * Tenants: $N\in\{20,100\}$
   * Private scaling: $a_i\sim Unif[0.8,1.6]$ (fixed per run).
   * Baseline price: $b_i=0.5$
   * Congestion rate: $\tau \in \{0.1,0.3,0.6\}$ (off-peak to peak)
   * Job size and time value: $w_i \sim Lognormal(\mu = 0, \sigma = 0.6), \gamma_i \in \{0.5,1,2\}$
   * Capacity and penalty: $C=0.95\cdot X_{free}, \quad \lambda =10$, where $X_{free}$ denotes the free total load when we solve the game once with $\lambda = 0$ using GP.
   * Lower bound: $\epsilon = 10^{-6}$

   Unless stated, numbers are fixed wityhin a run and resampled across runs.

2. Initialization  and Hyperparameters

   We compare BR and GP exactly as defined in previous section.

   BR:

   * set $\gamma^k=0.5$ when updating
   * solve the positive root of the equation via Newton(max 5 iterations, fallback to bisection).

   GP:

   * $\eta^k \in{10^{-3}, 5\times10^{-3}}$

   For initialization, set $x^0 = \epsilon\cdot(1+10^{-3}\xi_i), \xi_i \sim N(0,1)$.

   We set the stopping criteria as follows:

   * Stationarity: $\varepsilon_{stat} = 10^{-6}$
   * Max iterations: $K_{\max} = 2000$.

   For each $(N,\tau)$ tuple we run 5 random seeds and report the mean result

   Last but not least, check the action positivity and overload condition.

3. Metrics

   * Stationarity for convergence
   * wall-clock time for efficiency
   * allocation outcomes
   * fairness: variance and gini coefficient of the outcome; correlation between $x_i$and urgency $w_i\gamma_i$

4. Reproducibility and Environment:

   We use Python to simulate the game on AMD Ryzen 7 5800H CPU and 16GB Memory. We fixed random seeds per run and publish seeds and parameters for every figure. Also, all the code and scripts are available on github.

### Result

1. we illustrated the result with (N=100,tau=0.3,seed=2). it is representative because it has consifderable amount of nodes and intermediate congestion rate. We compare the algorithm in 3 aspects,as discussed in the methodology section.

   * stationairy vs iteration: 
     Figure 1 compares the stationarity measure over iterations for BR and GP at (N=100, τ=0.3). BR reaches the stationarity tolerance within XX iterations, whereas GP requires significantly more iterations.”

   * stationarity vs real timeline:

     (N=100,tau=0.3,seed=2) and (N=100,tau=0.1,seed=2) and(N=20,tau=0.3,seed=2)
     Although BR typically requires fewer iterations than GP to reach the same stationarity tolerance, we observe that GP can outperform BR in terms of wall-clock time when the system size NNN and the congestion parameter τ\tauτ are large. This is consistent with the per-iteration cost of the two methods.

     In our implementation, each BR iteration solves NNN one-dimensional cubic equations via a Newton–bisection procedure, which involves multiple function and derivative evaluations per player. Consequently, the computational cost of one BR step scales as O(N)O(N)O(N) with a relatively large constant factor. In contrast, each GP iteration only requires evaluating the gradient of the potential once and performing a single projected gradient update, which can be implemented as a fully vectorized O(N)O(N)O(N) operation with a small constant factor.
      As NNN grows, and as τ\tauτ increases (making the congestion term more dominant and the best-response cubic problems more ill-conditioned), the per-iteration cost of BR grows significantly, whereas the per-iteration cost of GP remains almost unchanged. This explains why, in the high-NNN, high-τ\tauτ regime, GP may require more iterations than BR but still achieves a shorter overall runtime.

     In short, BR is an iteration-efficient but iteration-expensive method, while GP is iteration-inefficient but iteration-cheap; in large, highly congested systems the second effect dominates.

   * fairness vs iteration
     “Figure 3 tracks the Gini coefficient of the allocation across iterations. Both algorithms quickly stabilize in terms of fairness; BR tends to yield slightly lower Gini on average (see also Table 1).”

2. Table 1 shows that BR converges reliably with fewer iterations, while achieving comparable or better fairness compared to GP.”

   



### Conclusion

Below is a draft **Conclusion** section you can drop into your paper and tweak as needed:

------

### 6. Conclusion

In this work we modelled a cloud resource allocation problem with heterogeneous users as a finite potential game. Each user chooses a unidimensional allocation variable that trades off local scaling cost, linear pricing, and delay sensitivity, while a shared congestion term and a quadratic penalty on capacity violations capture system-wide coupling. This formulation yields a smooth potential function whose stationary points coincide with the Nash equilibria of the game, and it provides a natural way to incorporate a soft capacity constraint via a simple quadratic term rather than an explicit hard constraint.

On the algorithmic side, we compared two representative learning dynamics: a synchronous best-response (BR) scheme and a projected gradient play (GP) method. The BR algorithm updates all players by solving their one-dimensional cubic best-response problems (via a Newton–bisection routine) and applying a relaxation step. The GP algorithm performs fully vectorized projected gradient steps on the potential, with a simple step-size rule and step clipping to handle the singular (1/x_i^2) term in the gradient. Both methods are purely decentralized in the sense that they only require aggregate statistics (through the sum of allocations) and local parameters.

Our numerical experiments, conducted over a range of system sizes (N \in {20, 100}) and congestion parameters (\tau \in {0.1, 0.3, 0.6}), show that both BR and GP effectively drive the stationarity measure to a small tolerance. In terms of iteration count, BR is clearly more efficient: it typically converges in a few tens of iterations, while GP may require several hundred or more. However, this picture changes when we consider wall-clock time. Each BR iteration solves (N) one-dimensional nonlinear equations, which makes the per-iteration cost relatively high and increasingly expensive as (N) and (\tau) grow. In contrast, each GP iteration reduces to a single vectorized gradient evaluation and projection. Consequently, in small and moderately congested systems BR is faster in wall-clock time, but in large, highly congested systems we observe that GP—despite using more iterations—can achieve a shorter overall runtime. In other words, BR is iteration-efficient but iteration-expensive, whereas GP is iteration-inefficient but iteration-cheap, and in the high-(N), high-(\tau) regime the latter effect dominates.

We also evaluated the quality of the resulting allocations through fairness metrics, including the variance and Gini coefficient of the allocations and the correlation between allocated resources and users’ urgency parameters. Both algorithms quickly stabilize these metrics as the learning process progresses, and the final solutions exhibit a clear positive correlation between urgency and allocated resources, indicating that more delay-sensitive users indeed receive more capacity. Across parameter regimes, BR tends to produce slightly lower Gini coefficients and variance on average, suggesting marginally more balanced allocations, while GP remains comparable in fairness when it converges to a similar stationarity level.

Overall, our results highlight a practical trade-off between algorithmic sophistication and scalability in potential-game-based resource allocation. Exact or near-exact best-response updates can significantly reduce the number of iterations needed for convergence, but their per-iteration cost can become prohibitive in large, highly congested systems. Simpler gradient-based dynamics, on the other hand, scale more gracefully with the number of users and congestion intensity, and can therefore dominate in terms of total runtime even when they require more iterations. An interesting direction for future work is to design hybrid or adaptive schemes that interpolate between BR and GP, for example by occasionally performing partial best-response refinements on top of gradient steps, or by adjusting the update rule based on observed curvature and congestion levels. Extending the analysis to time-varying demand, stochastic arrivals, and fully distributed implementations is another promising avenue to bridge the gap between the theoretical model and real-world cloud and edge computing systems.
