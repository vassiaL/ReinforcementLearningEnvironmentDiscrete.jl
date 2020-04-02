function getemptymaze(dimx, dimy)
    maze = ones(Int, dimx, dimy)
    maze[1,:] .= maze[end,:] .= 0
    maze[:, 1] .= maze[:, end] .= 0
    maze
end

function setwall!(maze, startpos, endpos)
    dimx, dimy = startpos - endpos
    if dimx == 0
        maze[startpos[1], startpos[2]:endpos[2]] .= 0
    else
        maze[startpos[1]:endpos[1], startpos[2]] .= 0
    end
end

function indto2d(maze, pos)
    dimx = size(maze, 1)
    [rem(pos, dimx), div(pos, dimx) + 1]
end
function posto1d(maze, pos)
    dimx = size(maze, 1)
    (pos[2] - 1) * dimx + pos[1]
end

function checkpos(maze, pos)
    count = 0
    for dx in -1:1
        for dy in -1:1
            count += maze[(pos + [dx, dy])...] == 0
        end
    end
    count
end

function addrandomwall!(maze)
    startpos = rand(ENV_RNG, findall(x -> x != 0, maze[:]))
    startpos = indto2d(maze, startpos)
    starttouch = checkpos(maze, startpos)
    if starttouch > 0
        return 0
    end
    endx, endy = startpos
    if rand(ENV_RNG, 0:1) == 0 # horizontal
        while checkpos(maze, [endx, startpos[2]]) == 0
            endx += 1
        end
        if maze[endx + 1, startpos[2]] == 1 &&
            maze[endx + 1, startpos[2] + 1] ==
            maze[endx + 1, startpos[2] - 1] == 0
            endx -= 1
        end
    else
        while checkpos(maze, [startpos[1], endy]) == 0
            endy += 1
        end
        if maze[startpos[1], endy + 1] == 1 &&
            maze[startpos[1] + 1, endy + 1] ==
            maze[startpos[1] - 1, endy + 1] == 0
            endx -= 1
        end
    end
    setwall!(maze, startpos, [endx, endy])
    return 1
end

function setTandR!(d)
    for s in d.statefrommaze[findall(x -> x != 0, reshape(d.maze, :))]
        setTandR!(d, s)
    end
end
function setTandR!(d, s)
    # @show s
    T = d.mdp.trans_probs
    R = d.mdp.reward
    statefrommaze = d.statefrommaze
    goals = d.goals
    # @show goals
    ns = d.mdp.observationspace.n
    maze = d.maze
    idx_goals = findfirst(x -> x == s, goals)
    # @show idx_goals
    # @show d.goalrewards
    #### TODO #####
    # Here R should be reset to 0s everwhere!!!!
    # Otherwise what you had before stay same!!!!!
    if idx_goals !== nothing
        # @show d.goalrewards[idx_goals]
        R.value[s] = d.goalrewards[idx_goals]
    end
    # @show unique(R.value)
    pos = indto2d(maze, d.mazefromstate[s])
    # @show pos
    if (!(in(:chosenactionweight, fieldnames(typeof(d))))
        || (d.chosenactionweight == 1.))
        for (aind, a) in enumerate(([0, 1], [1, 0], [0, -1], [-1, 0]))
            # @show (aind, a)
            nextpos = maze[(pos + a)...] == 0 ? pos : pos + a
            # @show nextpos
            if d.neighbourstateweight > 0
                positions = []
                push!(positions, nextpos)
                # @show positions
                weights = [1.]
                for dir in ([0, 1], [1, 0], [0, -1], [-1, 0])
                    if maze[(nextpos + dir)...] != 0
                        push!(positions, nextpos + dir)
                        push!(weights, d.neighbourstateweight)
                    end
                end
                # @show positions
                # @show weights
                states = map(p -> statefrommaze[posto1d(maze, p)], positions)
                weights /= sum(weights)
                # @show weights
                T[aind, s] = sparsevec(states, weights, ns)
            else
                nexts = statefrommaze[posto1d(maze, nextpos)]
                T[aind, s] = sparsevec([nexts], [1.], ns)
            end
        end
    else
        positions = []
        for (aind, a) in enumerate(([0, 1], [1, 0], [0, -1], [-1, 0]))
            # @show (aind, a)
            nextpos = maze[(pos + a)...] == 0 ? pos : pos + a
            push!(positions, nextpos)
        end
        # @show positions
        for (aind, a) in enumerate(([0, 1], [1, 0], [0, -1], [-1, 0]))
            # @show (aind, a)
            weights = (1. - d.chosenactionweight)/ 3. * ones(length(positions))
            weights[aind] = d.chosenactionweight
            weights /= sum(weights)
            # @show weights
            states = map(p -> statefrommaze[posto1d(maze, p)], positions)
            # @show states
            T[aind, s] = sparsevec(states, weights, ns)
            # @show T[aind, s]
        end
    end
end

function isinsideframe(maze, i)
    nx, ny = size(maze)
    i > nx && i < nx * (ny - 1) && i % nx != 0 && i % nx != 1
end
function n_effective(n, f, list)
    N = n === nothing ? div(length(list), Int(1/f)) : n
    min(N, length(list))
end
function breaksomewalls!(m; f = 1/50, n = nothing, rng = ENV_RNG)
    zeros = Int[]
    for i in 1:length(m)
        m[i] == 0 && isinsideframe(m, i) && push!(zeros, i)
    end
    pos = sample(rng, zeros, n_effective(n, f, zeros), replace = false)
    m[pos] .= 1
    m
end
export breaksomewalls!
function addobstacles!(m; f = 1/100, n = nothing, rng = ENV_RNG)
    nz = findall(x -> x == 1, reshape(m, :))
    pos = sample(rng, nz, n_effective(n, f, nz), replace = false)
    m[pos] .= 0
    m
end
export addobstacles!
"""
    struct DiscreteMaze
        mdp::MDP
        maze::Array{Int, 2}
        goals::Array{Int, 1}
        statefrommaze::Array{Int, 1}
        mazefromstate::Array{Int, 1}
"""
struct DiscreteMaze{T}
    mdp::T
    maze::Array{Int, 2}
    goals::Array{Int, 1}
    goalrewards::Array{Float64, 1}
    statefrommaze::Array{Int, 1}
    mazefromstate::Array{Int, 1}
    neighbourstateweight::Float64
    chosenactionweight::Float64
end
"""
    DiscreteMaze(; nx = 40, ny = 40, nwalls = div(nx*ny, 10), ngoals = 1,
                   goalrewards = 1, stepcost = 0, stochastic = false,
                   neighbourstateweight = .05)

Returns a `DiscreteMaze` of width `nx` and height `ny` with `nwalls` walls and
`ngoals` goal locations with reward `goalreward` (a list of different rewards
for the different goal states or constant reward for all goals), cost of moving
`stepcost` (reward = -`stepcost`); if `stochastic = true` the actions lead with
a certain probability to a neighbouring state, where `neighbourstateweight`
controls this probability.
"""
function DiscreteMaze(; nx = 40, ny = 40, nwalls = div(nx*ny, 10), kwargs...)
    m = getemptymaze(nx, ny)
    [addrandomwall!(m) for _ in 1:nwalls]
    breaksomewalls!(m)
    DiscreteMaze(m; kwargs...)
end
function DiscreteMaze(maze; ngoals = 1, goalrewards = 1., stepcost = 0,
                      stochastic = false,
                      neighbourstateweight = stochastic ? .05 : 0.,
                      compressed = true,
                      chosenactionweight = 1.)
    na = 4
    nzpos = findall(x -> x != 0, reshape(maze, :))
    statefrommaze = compressed ? cumsum(reshape(maze, :)) : collect(1:length(maze))
    mazefromstate = compressed ? nzpos : collect(1:length(maze))
    legalstates = statefrommaze[nzpos]
    ns = length(mazefromstate)
    T = Array{SparseVector{Float64,Int}}(undef, na, ns)
    goals = sort(sample(ENV_RNG, legalstates, ngoals, replace = false))
    R = DeterministicNextStateReward(fill(-stepcost, ns))
    isterminal = zeros(Int, ns); isterminal[goals] .= 1
    isinitial = setdiff(legalstates, goals)
    res = DiscreteMaze(MDP(DiscreteSpace(ns, 1),
                           DiscreteSpace(na, 1),
                           rand(ENV_RNG, legalstates),
                           T, R,
                           isinitial, isterminal),
                       maze,
                       goals,
                       typeof(goalrewards) <: Number ? fill(goalrewards, ngoals) :
                                                       goalrewards,
                       statefrommaze,
                       mazefromstate,
                       neighbourstateweight,
                       chosenactionweight)
    setTandR!(res)
    res
end

interact!(env::DiscreteMaze, a) = interact!(env.mdp, a)
reset!(env::DiscreteMaze) = reset!(env.mdp)
getstate(env::DiscreteMaze) = getstate(env.mdp)
actionspace(env::DiscreteMaze) = actionspace(env.mdp)

mutable struct ChangeDiscreteMaze{DiscreteMaze}
    discretemaze::DiscreteMaze
    stepcounter::Int
    switchsteps::Array{Int, 1}
    switchflag::Array{Bool, 2} # Used for RecordSwitches callback
    switchpos::Array{Int, 1}
    chosenactionweight::Float64
end
function ChangeDiscreteMaze(; switchsteps = [10^2], stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.,
                            nswitches = 1, chosenactionweight = 1.)
    dm = DiscreteMaze(neighbourstateweight = neighbourstateweight,
                        chosenactionweight = chosenactionweight)
    switchflag = Array{Bool, 2}(undef, 4, size(dm.maze, 1)*size(dm.maze, 2))
    switchflag .= false
    switchpos = rand(ENV_RNG, 1:length(reshape(dm.maze, :)), nswitches)
    ChangeDiscreteMaze(dm, 0, switchsteps, switchflag, switchpos, 1.)
end
function ChangeDiscreteMaze(maze; switchsteps = [10^2], stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.,
                            nswitches = 1, ngoals = 1,
                            chosenactionweight = 2. /3.)

    dm = DiscreteMaze(maze, ngoals = ngoals,
                        compressed = switchflag,
                        stochastic = stochastic,
                        neighbourstateweight = neighbourstateweight,
                        chosenactionweight = chosenactionweight)
    switchflag = Array{Bool, 2}(undef, 4, size(dm.maze, 1)*size(dm.maze, 2))
    switchflag .= false
    switchpos = rand(ENV_RNG, 1:length(reshape(dm.maze, :)), nswitches)
    ChangeDiscreteMaze(dm, 0, switchsteps, switchflag, switchpos, chosenactionweight)
end
function ChangeDiscreteMaze(maze, switchpos;
                            switchsteps = 10^2*ones(Int, length(switchpos)),
                            stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.,
                            ngoals = 1,
                            chosenactionweight = 2. /3.)

    dm = DiscreteMaze(maze, ngoals = ngoals,
                        compressed = false,
                        stochastic = stochastic,
                        neighbourstateweight = neighbourstateweight,
                        chosenactionweight = chosenactionweight)
    switchflag = Array{Bool, 2}(undef, 4, size(dm.maze, 1)*size(dm.maze, 2))
    switchflag .= false
    ChangeDiscreteMaze(dm, 0, switchsteps, switchflag, switchpos, chosenactionweight)
end

function interact!(env::ChangeDiscreteMaze, action)
    env.stepcounter += 1
    env.switchflag .= false
    #@show env.stepcounter
    if any(env.stepcounter .== env.switchsteps)
        # && (env.discretemaze.mdp.isterminal[env.discretemaze.mdp.state] == 1))# Switch or not!
        #println("####################################################################################################")
        nswitches = findall(env.stepcounter .== env.switchsteps)
        previousT = deepcopy(env.discretemaze.mdp.trans_probs)
        for i in nswitches
            setupswitch!(env, i)
        end
        updateswitchflag!(env, previousT)
    end
    interact!(env.discretemaze.mdp, action)
end
function setupswitch!(env::ChangeDiscreteMaze, iswitch)
    #for i in 1:length(env.switchpos)
    env.discretemaze.maze[env.switchpos[iswitch]] = 1 - env.discretemaze.maze[env.switchpos[iswitch]]
    #end
    # # !!!!! Reset the whole transition matrix, so that new walls become "undef"!!!
    # NOTE: Not important for normal RL learners. But important for MDPlearner!
    ns = env.discretemaze.mdp.observationspace.n
    na = env.discretemaze.mdp.actionspace.n
    env.discretemaze.mdp.trans_probs[:] = Array{SparseVector{Float64,Int}}(undef, na, ns)

    setTandR!(env.discretemaze)
end
function updateswitchflag!(env, previousT)
    for i in 1:length(env.switchflag)
        # @show i
        if !isassigned(previousT, i)
            if isassigned(env.discretemaze.mdp.trans_probs, i)
                env.switchflag[i] = true
                # @show i
                # @show env.discretemaze.mdp.trans_probs[i]
                # println("Case 1")
            end
        else
            if !isassigned(env.discretemaze.mdp.trans_probs, i)
                env.switchflag[i] = true
                # println("Case 2")
                # @show i
                # @show previousT[i]
            else
                if !all(previousT[i] .== env.discretemaze.mdp.trans_probs[i])
                    env.switchflag[i] = true
                    # println("Case 3")
                    # @show previousT[i]
                    # @show env.discretemaze.mdp.trans_probs[i]
                end
            end
        end
    end
end
reset!(env::ChangeDiscreteMaze) = reset!(env.discretemaze.mdp)
getstate(env::ChangeDiscreteMaze) = getstate(env.discretemaze.mdp)
actionspace(env::ChangeDiscreteMaze) = actionspace(env.discretemaze.mdp)
plotenv(env::ChangeDiscreteMaze) = plotenv(env.discretemaze)

mutable struct RandomChangeDiscreteMaze{DiscreteMaze}
    discretemaze::DiscreteMaze
    nbreak::Int
    nadd::Int
    changeprobability::Float64
    switchflag::Array{Bool, 2} # Used for RecordSwitches callback.
    seed::Any
    rng::MersenneTwister # Used only for switches!
    chosenactionweight::Float64
end
function RandomChangeDiscreteMaze(; nx = 20, ny = 20, ngoals = 4,
                    nwalls = 10, compressed = false,
                    stochastic = false,
                    neighbourstateweight = stochastic ? .05 : 0.,
                    chosenactionweight = 2. /3., nbreak = 2,
                    nadd = 4, changeprobability = 0.01, seed = 3)

    rng = MersenneTwister(seed)
    dm = DiscreteMaze(nx = nx, ny = ny, ngoals = ngoals, nwalls = nwalls,
                        compressed = compressed, stochastic = stochastic,
                        neighbourstateweight = neighbourstateweight,
                        chosenactionweight = chosenactionweight)
    switchflag = Array{Bool, 2}(undef, 4, nx*ny)
    switchflag .= false
    RandomChangeDiscreteMaze(dm, nbreak, nadd, changeprobability, switchflag,
                            seed, rng, chosenactionweight)
end
function RandomChangeDiscreteMaze(maze; ngoals = 1, nbreak = 2, nadd = 4,
                                changeprobability = 0.01,
                                chosenactionweight = 2. /3.,
                                seed = 3)
    rng = MersenneTwister(seed)
    dm = DiscreteMaze(maze, ngoals = ngoals, compressed = false,
                        chosenactionweight = chosenactionweight)
    switchflag = Array{Bool, 2}(undef, 4, size(dm.maze, 1)*size(dm.maze, 2))
    switchflag .= false
    RandomChangeDiscreteMaze(dm, nbreak, nadd, changeprobability, switchflag,
                            seed, rng, chosenactionweight)
end
function interact!(env::RandomChangeDiscreteMaze, action)
    env.switchflag .= false
    r = rand(env.rng)
    if r < env.changeprobability # Switch or not!
        previousT = deepcopy(env.discretemaze.mdp.trans_probs)
        setupswitch!(env)
        updateswitchflag!(env, previousT)
    end
    interact!(env.discretemaze.mdp, action)
end
function setupswitch!(env::RandomChangeDiscreteMaze)
    breaksomewalls!(env.discretemaze.maze, n = env.nbreak, rng = env.rng)
    addobstacles!(env.discretemaze.maze, n = env.nadd, rng = env.rng)
    setTandR!(env.discretemaze)
end

reset!(env::RandomChangeDiscreteMaze) = reset!(env.discretemaze.mdp)
getstate(env::RandomChangeDiscreteMaze) = getstate(env.discretemaze.mdp)
actionspace(env::RandomChangeDiscreteMaze) = actionspace(env.discretemaze.mdp)
plotenv(env::RandomChangeDiscreteMaze) = plotenv(env.discretemaze)
using Plots
function plotenv(env::DiscreteMaze)
    goals = env.goals
    mazefromstate = env.mazefromstate
    m = deepcopy(env.maze)
    m[mazefromstate[goals]] .= 3
    # for i in 1:length(goals)
    #     if env.goalrewards[i] > 1.
    #         m[mazefromstate[goals[i]]] = 4
    #     end
    # end
    m[mazefromstate[env.mdp.state]] = 2
    imshow(m, colormap = 21, size = (400, 400))
end
