from _rl_agent import rl_agent
from _dumb_agent_sa import dumb_agent as dumb_agent1
from _dumb_agent_ur import dumb_agent as dumb_agent2


if __name__ == "__main__":

    r1 = rl_agent()
    r1._simulator()

    r1 = dumb_agent1(fixed_action=0)
    r1._simulator()

    r1 = dumb_agent1(fixed_action=1)
    r1._simulator()

    r1 = dumb_agent2()
    r1._simulator()

