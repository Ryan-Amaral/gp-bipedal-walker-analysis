# gp-bipedal-walker-analysis
Results analysis of tests performed on the Box2D BipedalWalker environment for my Masters Thesis.

## Structure
`data/{x}/{y}` contains all the relevant data from a run, of run type `x` and run index `y`. Each contains the trainer, the final champion agent, and the generation data log.

`basic_analysis.ipynb` contains basic results obtained from the log files.

`trainer_analysis.ipynb` contains results obtained from analyzing the trainers in various ways, such as fitness distributions.

`champion_analysis.ipynb` contains results obtained from analyzing the champions in various ways, such as heatmapping learners used, and averaging champion run results over many episodes.
