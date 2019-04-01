# http://xgboost.readthedocs.io/en/latest/model.html

import numpy as np
from GameClass import Atomic_Splittable_Game

N_AGENTS = 2
N_RES = 2

avg_latency = [[] for _ in range(3)]
step_size_cntr = -1

for step_size in np.linspace(12.0, 18.0, 3):
    step_size_cntr += 1
    for perturbation_width in np.linspace(0.25, 0.35, 2):

        game = Atomic_Splittable_Game(N_AGENTS, N_RES)
        rsrc_object = game.get_rsrc_object()
        agent_list = game.get_agent_list()

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
                    agent.rsrc_history[rsrc].append(
                        (crnt_flow_matrix[agent_index,rsrc], crnt_latency[rsrc])
                    )

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

            for agent_index, agent in enumerate(agent_list):
                agent_flow = agent.randomize_optimal_flow(agent.get_opt_flow(), t)
                print "Agent", agent_index, " flow: ", agent_flow
                rsrc_object.set_flow_matrix(agent_flow, agent_index)

            crnt_flow_matrix = rsrc_object.crnt_flow_matrix
            crnt_latency = rsrc_object.get_latency_vector()
            total_latency.append(sum(crnt_latency) / float(len(crnt_latency)))

            if t % 5 == 0:
                print "Current latency vector", crnt_latency

            for agent_index, agent in enumerate(agent_list):
                for rsrc in range(N_RES):
                    agent.rsrc_history[rsrc].append(
                        (crnt_flow_matrix[agent_index,rsrc], crnt_latency[rsrc])
                    )
                agent.update_crnt_para_est()

        for agent_index, agent in enumerate(agent_list):
            print "Agent", agent_index, ": ", agent.para_est_a, agent.para_est_b

        avg_latency[step_size_cntr].append(total_latency[-1])

for list_i in avg_latency:
    print list_i