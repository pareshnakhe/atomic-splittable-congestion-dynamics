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
            #self.latency_func[i, :] = randNumGenerator.random_sample(2,)
            #self.latency_func[i, :] = np.array([(i+1)**2, (i+1)**2])
            self.latency_func[i, :] = np.array([(i+1)**3, 0])

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

        # parameters
        self.step_size = 10
        self.perturbation_width = 0.33

    def set_perturbation_width(self, width):
        self.perturbation_width = width

    def set_step_size(self, step_size):
        self.step_size = step_size

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
        # print "get_opt_flow:"
        # print self.para_est_a, self.para_est_b
        x_init = np.array([1.0 / N_RES for _ in range(N_RES)])
        result = optimize.minimize(
            lambda x: np.dot(self.para_est_a, x ** 2) + np.dot(self.para_est_b, x),
            x_init,
            constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}),
            bounds=([(0, 1) for _ in range(N_RES)]),
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


# By trial-and-error step-size 10-15 and perturbation width 0.33 seems to
# give the best results

# Main part begins
avg_latency = [[] for _ in range(3)]
step_size_cntr = -1
for step_size in np.linspace(12.0, 18.0, 3):
    step_size_cntr += 1
    for perturbation_width in np.linspace(0.25, 0.35, 2):

        rsrc_object = Resource()
        agent_list = [Agent() for _ in range(N_AGENTS)]

        for agent in agent_list:
            agent.set_step_size(step_size)
            agent.set_perturbation_width(perturbation_width)

        BOOT_RNDS = 1
        LEARN_RNDS = 51
        # Helper data structures for plotting
        total_latency = list()
        color_list = list()
        size_list = list()

        # Each agent chooses random flow and observes the corresponding latency.
        # No parameters are updates in this phase.
        for t in range(BOOT_RNDS):
            for agent_index, agent in enumerate(agent_list):
                agent_flow = agent.get_random_flow()
                rsrc_object.set_flow_matrix(agent_flow, agent_index)

            crnt_flow_matrix = rsrc_object.crnt_flow_matrix
            crnt_latency = rsrc_object.get_latency_vector()

            for agent_index, agent in enumerate(agent_list):
                for rsrc in range(N_RES):
                    agent.rsrc_history[rsrc].append((crnt_flow_matrix[agent_index,rsrc], crnt_latency[rsrc]))

            color_list.append('blue')
            size_list.append(50)


        # Use the random sampling data collected above
        # to generate first parameter estimates.
        for agent_index, agent in enumerate(agent_list):
            agent.update_crnt_para_est()


        # Now start with actual learning
        for t in range(LEARN_RNDS):
            color_list.append('red')
            size_list.append(25)

            # if t % 4 == 0:
            #     color_list.append('blue')
            #     size_list.append(40)
            # else:
            #     color_list.append('red')
            #     size_list.append(20)

            for agent_index, agent in enumerate(agent_list):
                agent_flow = agent.randomize_optimal_flow(agent.get_opt_flow(), t)

                # # exploration rounds
                # if t % 4 == 0:
                #     agent_flow = agent.get_random_flow()
                # # exploitation rounds
                # else:
                #     # get optimal flow for current parameter estimates
                #     agent_flow = agent.get_opt_flow()
                #     randomize_optimal_flow(agent_flow)
                #     exit(1)

                # if t % 2 == 0:
                print "Agent", agent_index, " flow: ", agent_flow
                rsrc_object.set_flow_matrix(agent_flow, agent_index)

            crnt_flow_matrix = rsrc_object.crnt_flow_matrix
            crnt_latency = rsrc_object.get_latency_vector()
            total_latency.append(sum(crnt_latency) / float(len(crnt_latency)))

            if t % 5 == 0:
                print "Current latency vector", crnt_latency

            for agent_index, agent in enumerate(agent_list):
                for rsrc in range(N_RES):
                    agent.rsrc_history[rsrc].append((crnt_flow_matrix[agent_index,rsrc], crnt_latency[rsrc]))
                agent.update_crnt_para_est()

        for agent_index, agent in enumerate(agent_list):
            print "Agent", agent_index, ": ", agent.para_est_a, agent.para_est_b

        avg_latency[step_size_cntr].append(total_latency[-1])
        testList = agent_list[0].rsrc_history[0]
        testList = [(math.exp(i), math.exp(j)) for (i,j) in testList]
        plt.subplot(211)
        plt.title('Agent[0]')
        plt.ylabel('latency of rsrc 0')
        plt.xlabel('flow in rsrc 0')
        # Idea: https://stackoverflow.com/questions/18458734/python-plot-list-of-tuples
        plt.scatter(*zip(*testList), c=color_list, s=size_list)

        plt.subplot(212)
        plt.ylabel('avg latency')
        plt.xlabel('time')
        plt.plot(total_latency)
        label = str(step_size) + "-" + str(perturbation_width) + ".pdf"
        print label
        plt.savefig(label)

print '\n'
for list_i in avg_latency:
    print list_i