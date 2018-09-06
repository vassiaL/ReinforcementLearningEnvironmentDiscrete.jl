using Test, ReinforcementLearningEnvironmentDiscrete

# write your own tests here
env = DiscreteMaze()
reset!(env)
@test typeof(interact!(env, 1)) == NamedTuple{(:observation, :reward, :isdone),
                                              Tuple{Int64,Float64,Bool}}
@test typeof(getstate(env)) == NamedTuple{(:observation, :isdone), 
                                          Tuple{Int64, Bool}}

using POMDPModels
env = POMDPEnv(TigerPOMDP())
reset!(env)
@test typeof(interact!(env, 1)) == NamedTuple{(:observation, :reward, :isdone),
                                              Tuple{Int64,Float64,Bool}}
@test typeof(getstate(env)) == NamedTuple{(:observation, :isdone), 
                                          Tuple{Int64, Bool}}
env = MDPEnv(GridWorld())
reset!(env)
@test typeof(interact!(env, 1)) == NamedTuple{(:observation, :reward, :isdone),
                                              Tuple{Int64,Float64,Bool}}
@test typeof(getstate(env)) == NamedTuple{(:observation, :isdone), 
                                          Tuple{Int64, Bool}}

env = CliffWalking()
reset!(env)
@test env.mdp.state == 1
