using Test, ReinforcementLearningEnvironmentDiscrete, POMDPModels,
ReinforcementLearningBase

for env in [MDP(), MDPEnv(GridWorld()), POMDPEnv(TigerPOMDP()), DiscreteMaze()]
    test_envinterface(env)
end

env = CliffWalking()
reset!(env)
@test env.mdp.state == 1
