using RLEnvDiscrete

env = DiscreteMaze(ngoals = 5)
rlsetup = RLSetup(SmallBackups(na = 4, ns = env.mdp.ns, γ = .99), 
                  env, ConstantNumberSteps(200), 
                  policy = EpsilonGreedyPolicy(.1),
                  callbacks = [VisualizeMaze(wait = 0)])
info("Before learning.") 
# run!(rlsetup)
rlsetup.callbacks = []
rlsetup.stoppingcriterion = ConstantNumberSteps(10^6)
learn!(rlsetup)
info("After learning.")
rlsetup.callbacks = [VisualizeMaze()]
rlsetup.stoppingcriterion = ConstantNumberSteps(100)
run!(rlsetup)
