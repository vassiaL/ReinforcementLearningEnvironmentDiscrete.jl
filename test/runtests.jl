using RLEnvDiscrete
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here
import RLEnvDiscrete: reset!, interact!, getstate
env = DiscreteMaze()
reset!(env)
@test typeof(interact!(1, env)) == Tuple{Int64,Float64,Bool}
@test typeof(getstate(env)) == Tuple{Int64, Bool}

using POMDPModels
env = POMDPEnvironment(TigerPOMDP())
reset!(env)
@test typeof(interact!(1, env)) == Tuple{Int64,Float64,Bool}
@test typeof(getstate(env)) == Tuple{Int64, Bool}
env = MDPEnvironment(GridWorld())
reset!(env)
@test typeof(interact!(1, env)) == Tuple{Int64,Float64,Bool}
@test typeof(getstate(env)) == Tuple{Int64, Bool}

env = CliffWalkingMDP()
reset!(env)
@test env.state == 1
