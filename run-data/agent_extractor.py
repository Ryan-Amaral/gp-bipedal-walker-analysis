from tpg.trainer import Trainer, loadTrainer
from tpg.agent import Agent

trainer = loadTrainer("trainer.pkl")

agent = trainer.getEliteAgent("BipedalWalker-v3")

agent.saveToFile("agent.pkl")
