using ReinforcementLearningEnvironmentDiscrete
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here
import ReinforcementLearningEnvironmentDiscrete: reset!, interact!, getstate
env = DiscreteMaze()
reset!(env)
@test typeof(interact!(1, env)) == Tuple{Int64,Float64,Bool}
@test typeof(getstate(env)) == Tuple{Int64, Bool}

# using POMDPModels
# env = POMDPEnvironment(TigerPOMDP())
# reset!(env)
# @test typeof(interact!(1, env)) == Tuple{Int64,Float64,Bool}
# @test typeof(getstate(env)) == Tuple{Int64, Bool}
# env = MDPEnvironment(GridWorld())
# reset!(env)
# @test typeof(interact!(1, env)) == Tuple{Int64,Float64,Bool}
# @test typeof(getstate(env)) == Tuple{Int64, Bool}

env = CliffWalking()
reset!(env)
@test env.mdp.state == 1
