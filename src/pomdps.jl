const rng = MersenneTwister(0)

mutable struct POMDPEnv{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
    actionspace::DiscreteSpace
end
POMDPEnv(model) = POMDPEnv(model, initialstate(model, rng), actions(model),
                           DiscreteSpace(n_actions(model), 1))

mutable struct MDPEnv{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
    actionspace::DiscreteSpace
end
MDPEnv(model) = MDPEnv(model, initialstate(model, rng), actions(model),
                       DiscreteSpace(n_actions(model), 1))

actionspace(env::Union{MDPEnv, POMDPEnv}) = env.actionspace
observation_index(env, o) = Int64(o) + 1

function interact!(env::POMDPEnv, action) 
    s, o, r = generate_sor(env.model, env.state, env.actions[action], rng)
    env.state = s
    (observation = observation_index(env.model, o), 
     reward = r, 
     isdone = isterminal(env.model, s))
end
function reset!(env::Union{POMDPEnv, MDPEnv})
    env.state = initialstate(env.model, rng)
    (observation = env.state)
end
function getstate(env::POMDPEnv)
    (observation = observation_index(env.model, generate_o(env.model, env.state, rng)),
     isdone = isterminal(env.model, env.state))
end

function interact!(env::MDPEnv, action)
    s = rand(rng, transition(env.model, env.state, env.actions[action]))
    r = reward(env.model, env.state, env.actions[action])
    env.state = s
    (observation = state_index(env.model, s), 
     reward = r, 
     isdone = isterminal(env.model, s))
end
function getstate(env::MDPEnv)
    (observation = state_index(env.model, env.state), 
     isdone = isterminal(env.model, env.state))
end

