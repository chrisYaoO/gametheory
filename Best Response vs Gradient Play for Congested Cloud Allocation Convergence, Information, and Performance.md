## Best Response vs Gradient Play for Congested Cloud Allocation: Convergence, Information, and Performance

Topic: Type 4 (d): continuous action games: Best-response (BR) and gradient play (GP)

Ruijie Yao chris.yao@mail.utoronto.ca

### Abstract

We study a repeated N-player continuous-action game that models cloud computational resource allocation under congestion and deadline pressure. Each tenant i chooses a non-negative and continuous resource rate (GPU-hours per unit time) to minimize a cost function that combines (i) self-side scaling cost with diminishing returns, (ii) a baseline price , (iii) a congestion externality that grows with total load and (iv) a time-loss penalty inversely proportional to allocated rate, capturing the incentive to finish sooner for jobs. The game comes with a strictly convex potential, guaranteeing a unique Nash equilibrium (NE). We compare two lightweight learning dynamics— Best Response (BR), implemented as block coordinate descent via 1D cubic solves, and Gradient Play (GP), a projected gradient descent on the potential. Simulations across heterogeneous job sizes, and capacity penalties evaluate convergence speed, wall-clock time, social cost and utilization. These results clarify the algorithmic trade-off between per-iteration accuracy and decentralized implementability, and suggest tuning congestion/price signals to steer the decentralized equilibrium toward efficient, time-aware allocations in multi-tenant clouds.

### Introduction



### Model & Classes of Games

In a cloud computing scenario, each player $P_i$ needs to choose a continuous and non-nagative resource rate $x_i$ for their own tasks, so that his cost is minimized. 

The cost considers those factors:

* time penalty for its task: $\gamma_i \frac{w_i}{x_i}$, where $\gamma_i$ is the time value and $w_i$ is the task scale

* personal cost with marginal payoff decrease: $\frac{a_i}{2}x_i^2$
* Basic price: $b_ix_i$
* congestion impact: $\tau X x_i$, where $X = \sum_j x_j$ is the overall system load and $\tau\gt0$ is the congestion rate(Intuition: the cluster is slower and more expensive when congestion is heavier)

Therefore we can derive the cost function for each player: $J_i(x_i,X) = \gamma_i \frac{w_i}{x_i} + \frac{a_i}{2}x_i^2+ b_ix_i+ \tau X x_i$

And the objective for each player is:
$$
\min J_i(x_i,X)\\
s.t.\quad x_i>\epsilon,\\
X = \sum_ix_i \le C
$$
The game now belongs to repeated N player continuous kernel game. We can use best response play(BR) or Gradient Play(GR) to update $x_i$ iteratively. In realization, the player doesn't have to know other's response, as long as he knows the overall system load $X$.





### 

### Algorithms

1. Best response Play

   

2. Gradient play

### Experiment Mehodology

### Result

### Conclusion

