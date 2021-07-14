from tpg.agent import Agent, loadAgent
import pickle
import gym
from numpy import append, mean, std, clip
from math import sin, cos, pi
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--reg_reset", "-r", action="store_true", dest="reg_reset", default=False)
(opts, args) = parser.parse_args()

"""
Function to run a single agent on the environment, compatible with multiprocessing.
"""
def run_agent(args):
    agent = args[0] # the agent
    env_name = args[1] # name of OpenAI environment
    episodes = args[2] # number of times to repeat game
    frames = args[3] # frames to play for

    agent.configFunctionsSelf()
    agent.zeroRegisters()

    scores = []
    
    #env = gym.make(env_name)

    for ep in range(episodes): # episode loop
    
        if opts.reg_reset:
            agent.zeroRegisters()

        env = gym.make(env_name)
        state = env.reset()
        score_ep = 0

        for i in range(frames): # frame loop

            state = append(state, [2*sin(0.2*pi*i), 2*cos(0.2*pi*i),
                               2*sin(0.1*pi*i), 2*cos(0.1*pi*i),
                               2*sin(0.05*pi*i), 2*cos(0.05*pi*i)])

            act = agent.act(state)[1]
            act = clip(act, -1, 1)

            # feedback from env
            state, reward, is_done, _ = env.step(act)

            score_ep += reward # accumulate reward in score
            if is_done:
                break # end early if losing state

        scores.append(score_ep)
        print(f"Ep: {ep}, Score: {score_ep}")

        env.close()

    final_score = mean(scores)

    agent.reward(scores, "Scores")
    agent.reward(final_score, "Mean")
    agent.reward(std(scores), "Std")
    
    print(f"Mean: {final_score}")
    print(f"Std: {std(scores)}")
    
    
    agent.reward(final_score, env_name)
    
if __name__ == "__main__":
    agnt = loadAgent("agent.pkl")

    run_agent((agnt, "BipedalWalker-v3", 1000, 99999))
    
    if opts.reg_reset:
        pickle.dump(agnt.team.outcomes["Scores"], open("agent-reset-scores-2.pkl", "wb"))
    else:
        pickle.dump(agnt.team.outcomes["Scores"], open("agent-scores.pkl", "wb"))

