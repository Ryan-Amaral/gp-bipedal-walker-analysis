from tpg.trainer import Trainer, loadTrainer
from tpg.agent import Agent
import gym
from numpy import append, mean, std, clip
from math import sin, cos, pi
import pickle

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
    trainer = loadTrainer("trainer.pkl")
    agents = trainer.getAgents(sortTasks=["BipedalWalker-v3"])[:5]

    best_agent = None
    best_mean = -999999

    for i in range(5):
        run_agent((agents[i], "BipedalWalker-v3", 100, 99999))
        
        if agents[i].team.outcomes["Mean"] > best_mean:
            best_agent = agents[i]
            best_mean = agents[i].team.outcomes["Mean"]
        
        
    print(f"Final Mean: {best_mean}")
    print(f"Final Std: {best_agent.team.outcomes['Std']}")
    best_agent.saveToFile(f"agent2.pkl")
    pickle.dump(best_agent.team.outcomes["Scores"], open("agent-scores-3.pkl", "wb"))
    
    
    
    
    
    
    
    
    
    
    
