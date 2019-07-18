mutable struct POMDPEnv{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
    actionspace::DiscreteSpace
end
POMDPEnv(model) = POMDPEnv(model, initialstate(model, ENV_RNG), actions(model),
                           DiscreteSpace(n_actions(model), 1))

mutable struct MDPEnv{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
    actionspace::DiscreteSpace
end
MDPEnv(model) = MDPEnv(model, initialstate(model, ENV_RNG), actions(model),
                       DiscreteSpace(n_actions(model), 1))

actionspace(env::Union{MDPEnv, POMDPEnv}) = env.actionspace
observationindex(env, o) = Int64(o) + 1

function interact!(env::POMDPEnv, action)
    s, o, r = generate_sor(env.model, env.state, env.actions[action], ENV_RNG)
    env.state = s
    (observation = observationindex(env.model, o),
     reward = r,
     isdone = isterminal(env.model, s))
end
function reset!(env::Union{POMDPEnv, MDPEnv})
    env.state = initialstate(env.model, ENV_RNG)
    (observation = env.state,)
end
function getstate(env::POMDPEnv)
    (observation = observationindex(env.model, generate_o(env.model, env.state, ENV_RNG)),
     isdone = isterminal(env.model, env.state))
end

function interact!(env::MDPEnv, action)
    s = rand(ENV_RNG, transition(env.model, env.state, env.actions[action]))
    r = POMDPs.reward(env.model, env.state, env.actions[action])
    env.state = s
    (observation = stateindex(env.model, s),
     reward = r,
     isdone = isterminal(env.model, s))
end
function getstate(env::MDPEnv)
    (observation = stateindex(env.model, env.state),
     isdone = isterminal(env.model, env.state))
end
