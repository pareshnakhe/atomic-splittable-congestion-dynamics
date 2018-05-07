# atomic-splittable-congestion-dynamics

Consider an atomic splittable congestion game where N players/agents send flow across M parallel edges. Each edge edge has an associated latency function and is unknown to the agents. This is an empirical experiment to test the effectiveness of linear regression-based prediction dynamics on the convergence to equilibrium.

Each player has a flow of 1 unit it wants to send over the edges. After choosing a vector of flows, it observes the corresponding latency on that edge. Note that this is a function of the total flow on that edge. It is assumed that all edges have linear latency functions. In this experiment, I attempt to predict the parameters of these linear functions using linear regression-like techniques.

In each round, each player computes the **optimal** flow based on the current parameter estimates. This optimal flow is then perturbed with Guassian noise and then sent over the network. The variance of this noise is decreased as the number of rounds incerease. For each edge, the player maintains a record of the flow sent and the latency observed on that edge. Based on this data, the parameters are updated every round using follow the regularized technique. Since the system is dynamic, the weight associated with each such data sample is decayed as a function of $t^{-1/3}$.

The performance of this approach is compared with that of vanilla multiplicative weight update method.


====================================================================================================

Conclusion:

This approach is effective in bringing the system fairly fast to a state close to equilibrium. However, there are several parameters which have been set mostly based on trial-and-error method. For example, when updating the parameter estimates of the resources, what should be the step size? At what rate should the weighting of older data decrease? It is quite likely that the convergence of the system is not fast enough because of sub-optimal parameter setting.

The multiplicative weight update on the other hand works extremely fast with step size $1/ \sqrt{t}$. I believe this is not surprising given provable theoretical guarantees.
====================================================================================================


Lesson:

estimate-then-optimize technique seems not work in multi-player settings. Especially, the older data that is used for estimation is usually outdated and hence leads to inaccurate predictions. Although, a more systematic theoretical study might help set the right parameters for better performance. In addition, the Hedge-style algorithms perform especially better given that the strategy set of the players, i.e. the flow vector belongs to the probability simplex; an ideal setting for Hedge algorithms. For pricing problems, it is unlikely that this approach would work.
