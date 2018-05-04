# http://xgboost.readthedocs.io/en/latest/model.html

import numpy as np
import math
from scipy import optimize
import matplotlib.pyplot as plt

N_AGENTS = 2
N_RES = 2


class Resource:
    """
    This encapslulates all the resources/edges in the system.
    For resource i, latency function is given as:
    latency_func[i,0]* x + latency_func[i,1]
    """
    def __init__(self):
        randNumGenerator = np.random.RandomState(272)

        self.latency_func = np.ones([2, N_RES])
        # set the latency function for each resource
        for i in range(N_RES):
            # self.latency_func[i, :] = randNumGenerator.random_sample(2,)
            # self.latency_func[i, :] = np.array([(i+1)**2, (i+1)**2])
            self.latency_func[i, :] = np.array([i+1, 0])

        self.crnt_flow_matrix = np.zeros([N_AGENTS, N_RES])
        print "In Resource constructor:", self.latency_func[:, 1]

    def get_latency_vector(self):
        """
        returns the latency vector based on total flow
        :return: latency vector corresponding to flow x on each resource
        """

        total_flow_list = [sum(self.crnt_flow_matrix[:, i]) for i in range(N_RES)]

        return np.array(
            [total_flow_list[i]*self.latency_func[i,0] + self.latency_func[i,1] for i in range(N_RES)]
        )

    def set_flow_matrix(self, flow_vector, agent):
        self.crnt_flow_matrix[agent, :] = flow_vector


class Agent:
    """ Agent is the class of players. Each agent has a fixed flow
        that it wants to send across N_RES parallel edges.
    """
    def __init__(self):
        """
        para_est_a/b is a vector of current parameter estimates for this agent Object
        """
        #self.para_est_a = np.random.random_sample(N_RES,)
        if rsrc_object is None:
            print "Agent class has no resource object"
            exit(1)

        self.para_est_a = np.ones(N_RES)
        # this is a constant for each agent
        self.para_est_b = rsrc_object.latency_func[:, 1]

        # this stores the flow sent and the resulting latency
        # observed for each resource
        self.rsrc_history = list([[] for i in range(N_RES)])

    @staticmethod
    def get_random_flow():
        unnormalized_flow = np.random.random_sample(N_RES)
        return unnormalized_flow / np.sum(unnormalized_flow)

    def get_opt_flow(self):
        """
        Given the latency parameter estimates, this function computes the
        optimal flow with respect to these estimates
        :return: vector x of flows on each edge so that sum(x) = 1.
        """
        x_init = np.array([1.0 / N_RES for i in range(N_RES)])
        result = optimize.minimize(
            lambda x: np.dot(self.para_est_a, x ** 2) + np.dot(self.para_est_b, x),
            x_init,
            constraints=({'type': 'eq', 'fun': lambda x: sum(x) - 1}),
            bounds=([(0, 1) for i in range(N_RES)]),
            method='SLSQP')

        return result.x

    def update_crnt_para_est(self):
        """
        for each resource compute parameters A and B s.t
        A_t, B_t = \argmin_{A, B} \sum_t (A * x_t + B - K_t)^2 (this is plain linear regression)
        where x_t corresponds to the flow sent by the agent on this resource
        K_t corresponds to the total latency observed by the agent.
        Note: In a given round, all agents see the same latency on a given resource
        """
        # helper data-structure for looping over each individual resource history
        crnt_rsrc_history = list()

        # history[0] = [(1,2), (1.5, 2.5), (0.8, 1.5)]
        # history[1] = [(1,2), (1.1, 2.5), (0.8, 1.5)]

        # HERE WE ONLY WANT TO OPTIMIZE THE COEFFICIENT OF X (A). B IS FIXED

        for i in range(N_RES):
            A_est = self.para_est_a[i]
            B = self.para_est_b[i]
            crnt_rsrc_history = self.rsrc_history[i]

            # \sum_t 1/\sqrt(t) (A*x_t + B - d_t)^2
            # The idea is to decrease the weight of older observations as 1/\sqrt(t).
            result = optimize.minimize(
                    lambda x: (sum([math.sqrt(1.0/(index+1))*(x*data[0] + B - data[1])**2 for index, data in enumerate(crnt_rsrc_history)]) + 15.0*((A_est - x)**2)),
                    np.array([0.1]),  # x_init
                    bounds=([(0, None)]),
                    method='SLSQP')

            self.para_est_a[i] = result.x
            # print "Resource", i, " parameters:", result.x


# Main part begins

rsrc_object = Resource()
agent_list = [Agent() for _ in range(N_AGENTS)]



BOOT_RNDS = 3
LEARN_RNDS = 151
color_list = list()
size_list = list()

# Each agent chooses random flow and observes the corresponding latency.
# No parameters are updates in this phase.
for t in range(BOOT_RNDS):
    for agent_index, agent in enumerate(agent_list):
        agent_flow = agent.get_random_flow()
        # if t % 2 == 0:
        #     print "Agent", agent_index, " flow: ", agent_flow
        rsrc_object.set_flow_matrix(agent_flow, agent_index)

    crnt_flow_matrix = rsrc_object.crnt_flow_matrix
    crnt_latency = rsrc_object.get_latency_vector()

    # if t % 5 == 0:
    #     print "Current latency vector", crnt_latency

    for agent_index, agent in enumerate(agent_list):
        for rsrc in range(N_RES):
            agent.rsrc_history[rsrc].append((crnt_flow_matrix[agent_index,rsrc], crnt_latency[rsrc]))

    color_list.append('blue')
    size_list.append(40)


# Use the random sampling data collected above
# to generate first parameter estimates.
for agent_index, agent in enumerate(agent_list):
    agent.update_crnt_para_est()

# Now start with actual learning
for t in range(LEARN_RNDS):
    if t % 4 == 0:
        color_list.append('blue')
        size_list.append(40)
    else:
        color_list.append('red')
        size_list.append(20)

    for agent_index, agent in enumerate(agent_list):
        # exploration rounds
        if t % 4 == 0:
            agent_flow = agent.get_random_flow()
        # exploitation rounds
        else:
            # get optimal flow for current parameter estimates
            agent_flow = agent.get_opt_flow()
        # if t % 2 == 0:
            print "Agent", agent_index, " flow: ", agent_flow
        rsrc_object.set_flow_matrix(agent_flow, agent_index)

    crnt_flow_matrix = rsrc_object.crnt_flow_matrix
    crnt_latency = rsrc_object.get_latency_vector()

    if t % 5 == 0:
        print "Current latency vector", crnt_latency

    for agent_index, agent in enumerate(agent_list):
        for rsrc in range(N_RES):
            agent.rsrc_history[rsrc].append((crnt_flow_matrix[agent_index,rsrc], crnt_latency[rsrc]))

        #print "Agent", agent_index, " history: ", agent.rsrc_history, '\n'
        #print "Agent", agent_index
        agent.update_crnt_para_est()

for agent_index, agent in enumerate(agent_list):
    print "Agent", agent_index, ": ", agent.para_est_a, agent.para_est_b


testList = agent_list[0].rsrc_history[0]
# Idea: https://stackoverflow.com/questions/18458734/python-plot-list-of-tuples
plt.scatter(*zip(*testList), c=color_list, s=size_list)
plt.show()
exit(1)