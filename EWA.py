import numpy as np
import math
from scipy import optimize
from GameClass import Resource

class Atomic_Splittable_Game_MW:

    def __init__(self, n_agents, n_res):
        self.N_AGENTS = n_agents
        self.N_RES = n_res
        self.__rsrc_object = Resource(n_agents, n_res)
        self.agent_list = [Agent_MW(n_res) for _ in range(n_agents)]

    def get_rsrc_object(self):
        return self.__rsrc_object

    def get_agent_list(self):
        return self.agent_list

class Agent_MW:
    """
    This class is based on the Hedge algorithm.
    For details: http://www.cs.princeton.edu/~rlivni/cos511/lectures/lect18.pdf
    """
    def __init__(self, N_RES):
        self.N_RES = N_RES
        self.wt_vec = np.ones(N_RES)
        self.step_size = 1.0
        self.rnd_no = 1

    def get_MW_flow(self):
        """
        Uses the weight vector to compute the flow (distribution)
        :return: flow vector such that \sum x_i = 1
        """
        return self.wt_vec / np.sum(self.wt_vec)

    def get_feedback_vec(self, cost_vec):
        """
        Observed delay/latency/cost is fed back to the algorithm.
        This is used to update the weight vector.
        :return:
        """
        self.step_size = 1.0 / math.sqrt(self.rnd_no)
        update_vec = np.array([np.exp(- self.step_size * cost_vec[i]) for i in range(self.N_RES)])
        self.wt_vec = np.multiply(update_vec, self.wt_vec)
        self.rnd_no += 1