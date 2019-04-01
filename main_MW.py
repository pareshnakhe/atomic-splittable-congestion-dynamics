from EWA import Atomic_Splittable_Game_MW

N_AGENTS = 2
N_RES = 2

game = Atomic_Splittable_Game_MW(N_AGENTS, N_RES)
rsrc_object = game.get_rsrc_object()
agent_list = game.get_agent_list()


BOOT_RNDS = 1
LEARN_RNDS = 11

# Now start with actual learning
for t in range(LEARN_RNDS):

    for agent_index, agent in enumerate(agent_list):
        agent_flow = agent.get_MW_flow()
        print "Agent", agent_index, " flow: ", agent_flow

        rsrc_object.set_flow_matrix(agent_flow, agent_index)

    crnt_latency = rsrc_object.get_latency_vector()
    # total_latency.append(sum(crnt_latency) / float(len(crnt_latency)))

    if t % 5 == 0:
        print "Current latency vector", crnt_latency

    for agent_index, agent in enumerate(agent_list):
        agent.get_feedback_vec(crnt_latency)
