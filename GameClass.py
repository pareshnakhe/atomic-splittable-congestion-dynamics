import numpy as np
import math
from scipy import optimize

class Atomic_Splittable_Game:

    def __init__(self, n_agents, n_res):
        self.N_AGENTS = n_agents
        self.N_RES = n_res
        self.__rsrc_object = Resource(n_agents, n_res)
        self.agent_list = [Agent(n_agents, n_res, self.__rsrc_object) for _ in range(n_agents)]

    def get_rsrc_object(self):
        return self.__rsrc_object

    def get_agent_list(self):
        return self.agent_list

class Resource:
    """
    This encapslulates all the resources/edges in the system.
    For resource i, latency function is given as:
    latency_func[i,0]* x + latency_func[i,1]
    """
    def __init__(self, N_AGENTS, N_RES):
        self.N_AGENTS = N_AGENTS
        self.N_RES = N_RES
        randNumGenerator = np.random.RandomState(272)

        self.latency_func = np.ones([2, self.N_RES])
        # set the latency function for each resource
        for i in range(self.N_RES):
            #self.latency_func[i, :] = randNumGenerator.random_sample(2,)
            #self.latency_func[i, :] = np.array([(i+1)**2, (i+1)**2])
            self.latency_func[i, :] = np.array([(i+1)**3, 0])

        self.crnt_flow_matrix = np.zeros([self.N_AGENTS, self.N_RES])
        print "In Resource constructor:", self.latency_func[:, 1]

    def get_latency_vector(self):
        """
        returns the latency vector based on total flow
        :return: latency vector corresponding to flow x on each resource
        """

        total_flow_list = [sum(self.crnt_flow_matrix[:, i]) for i in range(self.N_RES)]

        return np.array(
            [total_flow_list[i]*self.latency_func[i,0] + self.latency_func[i,1] for i in range(self.N_RES)]
        )

    def set_flow_matrix(self, flow_vector, agent):
        """
        :param flow_vector: vector containing flow on each resource
        :param agent: associated agent
        :return: sets the crnt_flow_matrix values
        """
        self.crnt_flow_matrix[agent, :] = flow_vector


class Agent:
    """ Agent is the class of players. Each agent has a fixed flow
        that it wants to send across N_RES parallel edges.
    """
    def __init__(self, N_AGENTS, N_RES, rsrc_object):
        """
        para_est_a/b is a vector of current parameter estimates for this agent Object
        """
        self.N_AGENTS = N_AGENTS
        self.N_RES = N_RES
        self.rsrc_object = rsrc_object

        #self.para_est_a = np.random.random_sample(N_RES,)
        if self.rsrc_object is None:
            print "Agent class has no resource object"
            exit(1)

        self.para_est_a = np.ones(self.N_RES)
        # this is a constant for each agent
        self.para_est_b = self.rsrc_object.latency_func[:, 1]

        # this stores the flow sent and the resulting latency
        # observed for each resource
        self.rsrc_history = list([[] for i in range(self.N_RES)])

        # parameters
        self.step_size = 10
        self.perturbation_width = 0.33

    def set_perturbation_width(self, width):
        self.perturbation_width = width

    def set_step_size(self, step_size):
        self.step_size = step_size

    def get_random_flow(self):
        unnormalized_flow = np.random.random_sample(self.N_RES)
        return unnormalized_flow / np.sum(unnormalized_flow)

    def get_opt_flow(self):
        """
        Given the latency parameter estimates, this function computes the
        optimal flow with respect to these estimates
        :return: vector x of flows on each edge so that sum(x) = 1.
        """
        # print "get_opt_flow:"
        # print self.para_est_a, self.para_est_b
        x_init = np.array([1.0 / self.N_RES for _ in range(self.N_RES)])
        result = optimize.minimize(
            lambda x: np.dot(self.para_est_a, x ** 2) + np.dot(self.para_est_b, x),
            x_init,
            constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}),
            bounds=([(0, 1) for _ in range(self.N_RES)]),
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

        for i in range(self.N_RES):
            A_est = self.para_est_a[i]
            B = self.para_est_b[i]
            crnt_rsrc_history = self.rsrc_history[i]

            # \sum_t 1/\sqrt(t) (A*x_t + B - d_t)^2
            # The idea is to decrease the weight of older observations as 1/\sqrt(t).
            result = optimize.minimize(
                    lambda x: (sum([math.sqrt(1.0/(index+1))*(x*data[0] + B - data[1])**2 for index, data in enumerate(crnt_rsrc_history)]) + self.step_size*((A_est - x)**2)),
                    np.array([0.1]),  # x_init
                    bounds=([(0, None)]),
                    method='SLSQP')

            self.para_est_a[i] = result.x

    def randomize_optimal_flow(self, opt_flow, rnd):
        # print "randomize_optimal_flow:"
        rand_flow = list()
        for flow in opt_flow:
            # print flow
            mean = flow
            std = (flow / 3.0)
            # 0.33 BELOW IS JUST A PARAMETER. NOT SURE WHAT IS THE RIGHT VALUE
            rand_flow.append(np.random.normal(loc=mean, scale=std * math.pow(rnd + 1, -1 * self.perturbation_width)))

        rand_flow = rand_flow + opt_flow
        return rand_flow / sum(rand_flow)
