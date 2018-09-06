const rng = MersenneTwister(0)

mutable struct POMDPEnvironment{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
end
POMDPEnvironment(model) = POMDPEnvironment(model, 
                                           initialstate(model, rng),
                                           actions(model))
export POMDPEnvironment
mutable struct MDPEnvironment{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
end
MDPEnvironment(model) = MDPEnvironment(model,
                                       initialstate(model, rng),
                                       actions(model))
export MDPEnvironment

observation_index(env, o) = Int64(o) + 1

function interact!(env::POMDPEnvironment, action) 
    s, o, r = generate_sor(env.model, env.state, env.actions[action], rng)
    env.state = s
    (observation = observation_index(env.model, o), 
     reward = r, 
     isdone = isterminal(env.model, s))
end
function reset!(env::Union{POMDPEnvironment, MDPEnvironment})
    env.state = initialstate(env.model, rng)
    (observation = env.state)
end
function getstate(env::POMDPEnvironment)
    (observation = observation_index(env.model, generate_o(env.model, env.state, rng)),
     isdone = isterminal(env.model, env.state))
end

function interact!(env::MDPEnvironment, action)
    s = rand(rng, transition(env.model, env.state, env.actions[action]))
    r = reward(env.model, env.state, env.actions[action])
    env.state = s
    (observation = state_index(env.model, s), 
     reward = r, 
     isdone = isterminal(env.model, s))
end
function getstate(env::MDPEnvironment)
    (observation = state_index(env.model, env.state), 
     isdone = isterminal(env.model, env.state))
end

