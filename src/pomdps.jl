using POMDPModels

const rng = MersenneTwister(0)

type POMDPEnvironment{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
end
POMDPEnvironment(model) = POMDPEnvironment(model, 
                                           initial_state(model, rng),
                                           actions(model))
export POMDPEnvironment
type MDPEnvironment{T,Ts,Ta}
    model::T
    state::Ts
    actions::Ta
end
MDPEnvironment(model) = MDPEnvironment(model,
                                       initial_state(model, rng),
                                       actions(model))
export MDPEnvironment

observation_index(env, o) = Int64(o) + 1

function interact!(action, env::POMDPEnvironment) 
    s, o, r = generate_sor(env.model, env.state, env.actions[action], rng)
    env.state = s
    observation_index(env.model, o), r, isterminal(env.model, s)
end
function reset!(env::Union{POMDPEnvironment, MDPEnvironment})
    env.state = initial_state(env.model, rng)
    nothing
end
function getstate(env::POMDPEnvironment)
    observation_index(env.model, generate_o(env.model, env.state, rng)),
    isterminal(env.model, env.state)
end

function interact!(action, env::MDPEnvironment)
    s = rand(rng, transition(env.model, env.state, env.actions[action]))
    r = reward(env.model, env.state, env.actions[action])
    env.state = s
    state_index(env.model, s), r, isterminal(env.model, s)
end
function getstate(env::MDPEnvironment)
    state_index(env.model, env.state), isterminal(env.model, env.state)
end

