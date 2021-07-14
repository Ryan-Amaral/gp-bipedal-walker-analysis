from tpg.trainer import Trainer, loadTrainer
from tpg.agent import Agent

trainer = loadTrainer("trainer.pkl")

agent = trainer.getAgents(sortTasks=["BipedalWalker-v3"])[1]

agent.saveToFile("agent.pkl")
