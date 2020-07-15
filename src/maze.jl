using SparseArrays, LinearAlgebra
struct Maze
    dimx::Int
    dimy::Int
    walls::Symmetric{Bool,SparseMatrixCSC{Bool,Int}}
end
Base.length(m::Maze) = m.dimx * m.dimy
Base.size(m::Maze) = (m.dimx, m.dimy)
function Maze(; nx = 40, ny = 40, nwalls = div(nx*ny, 20), rng = ENV_RNG) #Random.GLOBAL_RNG)
    m = emptymaze(nx, ny)
    for _ in 1:nwalls
        addrandomwall!(m, rng = rng)
    end
    breaksomewalls!(m, rng = rng)
end
function emptymaze(dimx, dimy)
    Maze(dimx, dimy, Symmetric(sparse([], [], Bool[], dimx * dimy, dimx * dimy)))
end
iswall(maze, pos, dir) = hitborder(maze, pos, dir) || maze.walls[pos, nextpos(maze, pos, dir)]
function setwall!(maze, pos, dir)
    npos = nextpos(maze, pos, dir)
    maze.walls.data[sort([pos, npos])...] = true
    maze
end
function breakwall!(maze, pos, dir)
    npos = nextpos(maze, pos, dir)
    maze.walls.data[sort([pos, npos])...] = false
    dropzeros!(maze.walls.data)
    maze
end
function nextpos(maze, pos, dir)
    dir == :up && return pos - 1
    dir == :down && return pos + 1
    dir == :left && return pos - maze.dimy
    dir == :right && return pos + maze.dimy
end
function hitborder(maze, pos, dir)
    dir == :up && pos % maze.dimy == 1 && return true
    dir == :down && pos % maze.dimy == 0 && return true
    dir == :left && pos <= maze.dimy && return true
    dir == :right && pos > maze.dimy * (maze.dimx - 1) && return true
    return false
end

function orthogonal_directions(dir)
    dir ∈ (:up, :down) && return (:left, :right)
    return (:up, :down)
end

function is_wall_neighbour(maze, pos)
    for dir in (:up, :down, :left, :right)
        iswall(maze, pos, dir) && return true
    end
    for npos in (nextpos(maze, pos, :up), nextpos(maze, pos, :down))
        for dir in (:left, :right, :up, :down)
            iswall(maze, npos, dir) && return true
        end
    end
    return false
end
function is_wall_tangential(maze, pos, dir)
    for ortho_dir in orthogonal_directions(dir)
        iswall(maze, pos, ortho_dir) && return true
    end
    return false
end
is_wall_ahead(maze, pos, dir) = iswall(maze, pos, dir)

function addrandomwall!(maze; rng = ENV_RNG) # Random.GLOBAL_RNG)
    potential_startpos = filter(x -> !is_wall_neighbour(maze, x), 1:maze.dimx * maze.dimy)
    if potential_startpos == []
        @warn("Cannot add a random wall.")
        return maze
    end
    pos = rand(rng, potential_startpos)
    direction = rand(rng, (:up, :down, :left, :right))
    while true
        setwall!(maze, pos, orthogonal_directions(direction)[2])
        pos = nextpos(maze, pos, direction)
        is_wall_tangential(maze, pos, direction) && break
        if is_wall_ahead(maze, pos, direction)
            setwall!(maze, pos, orthogonal_directions(direction)[2])
            break
        end
    end
    return maze
end

function n_effective(n, f, list)
    N = n === nothing ? div(length(list), Int(1/f)) : n
    min(N, length(list))
end
function breaksomewalls!(m; f = 1/50, n = nothing, rng = ENV_RNG) #Random.GLOBAL_RNG)
    wall_idxs = findall(m.walls.data)
    pos = sample(rng, wall_idxs, n_effective(n, f, wall_idxs), replace = false)
    m.walls.data[pos] .= false
    dropzeros!(m.walls.data)
    m
end
# function addobstacles!(m; f = 1/100, n = nothing, rng = Random.GLOBAL_RNG)
#     nz = findall(x -> x == 1, reshape(m, :))
#     pos = sample(rng, nz, n_effective(n, f, nz), replace = false)
#     m[pos] .= 0
#     m
# end
setTandR!(d) = for s in 1:length(d.maze) setTandR!(d, s) end
function setTandR!(d, s)
    # @show s
    T = d.mdp.trans_probs
    R = d.mdp.reward
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
    if (!(in(:chosenactionweight, fieldnames(typeof(d))))
        || (d.chosenactionweight == 1.))
        for (aind, a) in enumerate((:up, :down, :left, :right))
            # @show (aind, a)
            npos = iswall(maze, s, a) ? s : nextpos(maze, s, a)
            # @show npos
            if d.neighbourstateweight > 0
                positions = [npos]
                # @show positions
                weights = [1.]
                for dir in (:up, :down, :left, :right)
                    if !iswall(maze, npos, dir)
                        push!(positions, nextpos(maze, npos, dir))
                        push!(weights, d.neighbourstateweight)
                    end
                end
                # @show positions
                # @show weights
                weights /= sum(weights)
                # @show weights
                T[aind, s] = sparsevec(positions, weights, ns)
            else
                T[aind, s] = sparsevec([npos], [1.], ns)
            end
        end
    else
        positions = []
        for (aind, a) in enumerate((:up, :down, :left, :right))
            # @show (aind, a)
            npos = iswall(maze, s, a) ? s : nextpos(maze, s, a)
            push!(positions, npos)
        end
        # @show positions
        for (aind, a) in enumerate((:up, :down, :left, :right))
            # @show (aind, a)
            weights = (1. - d.chosenactionweight)/ 3. * ones(length(positions))
            weights[aind] = d.chosenactionweight
            weights /= sum(weights)
            # @show weights
            T[aind, s] = sparsevec(positions, weights, ns)
            # @show T[aind, s]
        end
    end
end

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
    maze::Maze
    goals::Array{Int, 1}
    goalrewards::Array{Float64, 1}
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
DiscreteMaze(maze; kwargs...) = DiscreteMaze(; maze = maze, kwargs...)
function DiscreteMaze(; nx = 40, ny = 40, nwalls = div(nx*ny, 20),
                      rng = ENV_RNG, #Random.GLOBAL_RNG,
                      maze = Maze(nx = nx, ny = ny, nwalls = nwalls, rng = rng),
                      ngoals = 1,
                      goalrewards = 1.,
                      stepcost = 0,
                      stochastic = false,
                      neighbourstateweight = stochastic ? .05 : 0.,
                      chosenactionweight = 1.)
    na = 4
    ns = length(maze)
    legalstates = 1:ns
    T = Array{SparseVector{Float64,Int}}(undef, na, ns)
    goals = sort(sample(rng, legalstates, ngoals, replace = false))
    R = DeterministicNextStateReward(fill(-stepcost, ns))
    isterminal = zeros(Int, ns); isterminal[goals] .= 1
    isinitial = setdiff(legalstates, goals)
    res = DiscreteMaze(MDP(DiscreteSpace(ns, 1),
                           DiscreteSpace(na, 1),
                           rand(rng, legalstates),
                           T, R,
                           isinitial,
                           isterminal),
                       maze,
                       goals,
                       typeof(goalrewards) <: Number ? fill(goalrewards, ngoals) :
                                                       goalrewards,
                       neighbourstateweight,
                       chosenactionweight)
    setTandR!(res)
    res
end
interact!(env::DiscreteMaze, a) = interact!(env.mdp, a)
reset!(env::DiscreteMaze) = reset!(env.mdp)
getstate(env::DiscreteMaze) = getstate(env.mdp)
actionspace(env::DiscreteMaze) = actionspace(env.mdp)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
mutable struct ChangeDiscreteMaze{DiscreteMaze}
    discretemaze::DiscreteMaze
    stepcounter::Int
    switchsteps::Array{Int, 1}
    switchpos::Array{Int, 1}
    switchdir::Array{Symbol, 1}
    switchflag::Array{Bool, 2} # Used for RecordSwitches callback
    chosenactionweight::Float64
end
function ChangeDiscreteMaze(; nx = 40, ny = 40, nwalls = div(nx*ny, 20),
                            rng = ENV_RNG, #Random.GLOBAL_RNG,
                            maze = Maze(nx = nx, ny = ny, nwalls = nwalls, rng = rng),
                            stochastic = false,
                            ngoals = 1,  goalrewards = 1.,
                            nswitches = 1,
                            switchpos = rand(ENV_RNG, 1:length(maze), nswitches),
                            switchdir = rand(ENV_RNG, [:up, :down, :left, :right], nswitches),
                            switchsteps = 10^2 * ones(Int, length(switchpos)),
                            neighbourstateweight = stochastic ? .05 : 0.,
                            chosenactionweight = 1.)
    dm = DiscreteMaze(maze = maze, ngoals = ngoals, goalrewards=goalrewards,
                        neighbourstateweight = neighbourstateweight,
                        chosenactionweight = chosenactionweight)
    switchflag = fill(false, 4, length(dm.maze))
    ChangeDiscreteMaze(dm, 0, switchsteps, switchpos, switchdir, switchflag, chosenactionweight)
end
ChangeDiscreteMaze(maze; kwargs...) = ChangeDiscreteMaze(; maze = maze, kwargs...)
ChangeDiscreteMaze(maze, switchpos, switchdir; kwargs...) =
    ChangeDiscreteMaze(; maze = maze, switchpos = switchpos, switchdir = switchdir, kwargs...)
# --- For standard RL learners all we need is:
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
    if iswall(env.discretemaze.maze, env.switchpos[iswitch], env.switchdir[iswitch])
        breakwall!(env.discretemaze.maze, env.switchpos[iswitch], env.switchdir[iswitch])
    else
        setwall!(env.discretemaze.maze, env.switchpos[iswitch], env.switchdir[iswitch])
    end
    # env.discretemaze.maze[env.switchpos[iswitch]] = 1 - env.discretemaze.maze[env.switchpos[iswitch]]
    setTandR!(env.discretemaze)
end
# # --- For MDP learner we need this:
# function interact!(env::ChangeDiscreteMaze, action)
#     env.stepcounter += 1
#     env.switchflag .= false
#     #@show env.stepcounter
#     if any(env.stepcounter .== env.switchsteps)
#         previousT = deepcopy(env.discretemaze.mdp.trans_probs)
#         # && (env.discretemaze.mdp.isterminal[env.discretemaze.mdp.state] == 1))# Switch or not!
#         #println("###########################################################")
#         nswitches = findall(env.stepcounter .== env.switchsteps)
#         # @show env.switchsteps
#         # @show env.switchpos
#         if any(env.switchpos .== env.discretemaze.mdp.state)
#             for i in nswitches
#                 push!(env.switchsteps, env.stepcounter+1)
#                 push!(env.switchpos, env.switchpos[i])
#             end
#             # @show env.switchsteps
#             # @show env.switchpos
#             # println("i saw it!")
#         else
#             for i in nswitches
#                 # @show i
#                 setupswitch!(env, i)
#             end
#         end
#         updateswitchflag!(env, previousT)
#     end
#     interact!(env.discretemaze.mdp, action)
# end
# function setupswitch!(env::ChangeDiscreteMaze, iswitch)
#     #for i in 1:length(env.switchpos)
#     env.discretemaze.maze[env.switchpos[iswitch]] = 1 - env.discretemaze.maze[env.switchpos[iswitch]]
#     # @show env.discretemaze.maze
#     #end
#     # # !!!!! Reset the whole transition matrix, so that new walls become "undef"!!!
#     # NOTE: Not important for normal RL learners. But important for MDPlearner!
#     ns = env.discretemaze.mdp.observationspace.n
#     na = env.discretemaze.mdp.actionspace.n
#     env.discretemaze.mdp.trans_probs[:] = Array{SparseVector{Float64,Int}}(undef, na, ns)
#
#     setTandR!(env.discretemaze)
# end
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

mutable struct ChangeDiscreteMazeProbabilistic{DiscreteMaze}
    discretemaze::DiscreteMaze
    changeprobability::Float64
    switchpos::Array{Int, 1}
    switchdir::Array{Symbol, 1}
    switchflag::Array{Bool, 2} # Used for RecordSwitches callback
    chosenactionweight::Float64
    seed::Any
    rng::MersenneTwister # Used only for switches!
end
function ChangeDiscreteMazeProbabilistic(; nx = 40, ny = 40, nwalls = div(nx*ny, 20),
                            seed = 3,
                            rng = MersenneTwister(seed),
                            maze = Maze(nx = nx, ny = ny, nwalls = nwalls, rng = rng),
                            ngoals = 1,  goalrewards = 1.,
                            changeprobability = .01,
                            nswitches = 1,
                            switchpos = rand(ENV_RNG, 1:length(maze), nswitches),
                            switchdir = rand(ENV_RNG, [:up, :down, :left, :right], nswitches),
                            stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.,
                            chosenactionweight = 1.)
    dm = DiscreteMaze(maze = maze, ngoals = ngoals, goalrewards=goalrewards,
                        neighbourstateweight = neighbourstateweight,
                        chosenactionweight = chosenactionweight)
    switchflag = fill(false, 4, length(dm.maze))
    ChangeDiscreteMazeProbabilistic(dm, changeprobability, switchpos, switchdir,
                            switchflag, chosenactionweight, seed, rng)
end
ChangeDiscreteMazeProbabilistic(maze; kwargs...) =
    ChangeDiscreteMazeProbabilistic(; maze = maze, kwargs...)
ChangeDiscreteMazeProbabilistic(maze, switchpos, switchdir; kwargs...) =
    ChangeDiscreteMazeProbabilistic(; maze = maze, switchpos = switchpos, switchdir = switchdir, kwargs...)

# --- For standard RL learners all we need is:
function interact!(env::ChangeDiscreteMazeProbabilistic, action)
    env.switchflag .= false
    r = rand(env.rng)

    if r < env.changeprobability # Switch or not!
        # println("Switch!")
        # @show env.switchflag[:, 76]
        # @show env.discretemaze.mdp.trans_probs[1, 76]
        nswitches = length(env.switchpos)
        previousT = deepcopy(env.discretemaze.mdp.trans_probs)
        for i in nswitches
            setupswitch!(env, i)
        end
        updateswitchflag!(env, previousT)
        # @show env.switchflag[:, 76]
        # @show env.discretemaze.mdp.trans_probs[1, 76]
        nswitches = length(env.switchpos)
    end
    interact!(env.discretemaze.mdp, action)
end
# # --- For MDP learner we need this:
# function interact!(env::ChangeDiscreteMaze, action)
#     env.stepcounter += 1
#     env.switchflag .= false
#     #@show env.stepcounter
#     if any(env.stepcounter .== env.switchsteps)
#         previousT = deepcopy(env.discretemaze.mdp.trans_probs)
#         # && (env.discretemaze.mdp.isterminal[env.discretemaze.mdp.state] == 1))# Switch or not!
#         #println("###########################################################")
#         nswitches = findall(env.stepcounter .== env.switchsteps)
#         # @show env.switchsteps
#         # @show env.switchpos
#         if any(env.switchpos .== env.discretemaze.mdp.state)
#             for i in nswitches
#                 push!(env.switchsteps, env.stepcounter+1)
#                 push!(env.switchpos, env.switchpos[i])
#             end
#             # @show env.switchsteps
#             # @show env.switchpos
#             # println("i saw it!")
#         else
#             for i in nswitches
#                 # @show i
#                 setupswitch!(env, i)
#             end
#         end
#         updateswitchflag!(env, previousT)
#     end
#     interact!(env.discretemaze.mdp, action)
# end
# function setupswitch!(env::ChangeDiscreteMaze, iswitch)
#     #for i in 1:length(env.switchpos)
#     env.discretemaze.maze[env.switchpos[iswitch]] = 1 - env.discretemaze.maze[env.switchpos[iswitch]]
#     # @show env.discretemaze.maze
#     #end
#     # # !!!!! Reset the whole transition matrix, so that new walls become "undef"!!!
#     # NOTE: Not important for normal RL learners. But important for MDPlearner!
#     ns = env.discretemaze.mdp.observationspace.n
#     na = env.discretemaze.mdp.actionspace.n
#     env.discretemaze.mdp.trans_probs[:] = Array{SparseVector{Float64,Int}}(undef, na, ns)
#
#     setTandR!(env.discretemaze)
# end
function setupswitch!(env::ChangeDiscreteMazeProbabilistic, iswitch)
    if iswall(env.discretemaze.maze, env.switchpos[iswitch], env.switchdir[iswitch])
        breakwall!(env.discretemaze.maze, env.switchpos[iswitch], env.switchdir[iswitch])
    else
        setwall!(env.discretemaze.maze, env.switchpos[iswitch], env.switchdir[iswitch])
    end
    # env.discretemaze.maze[env.switchpos[iswitch]] = 1 - env.discretemaze.maze[env.switchpos[iswitch]]
    setTandR!(env.discretemaze)
end
reset!(env::ChangeDiscreteMazeProbabilistic) = reset!(env.discretemaze.mdp)
getstate(env::ChangeDiscreteMazeProbabilistic) = getstate(env.discretemaze.mdp)
actionspace(env::ChangeDiscreteMazeProbabilistic) = actionspace(env.discretemaze.mdp)
plotenv(env::ChangeDiscreteMazeProbabilistic) = plotenv(env.discretemaze)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# NOTE: RandomChangeDiscreteMaze is not tested
# ------------------------------------------------------------------------------
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
function RandomChangeDiscreteMaze(; nx = 20, ny = 20, nwalls = 10,
                                    seed = 3, rng = MersenneTwister(seed),
                            maze = Maze(nx = nx, ny = ny, nwalls = nwalls, rng = rng),
                            ngoals = 4, goalrewards = 1.,
                    stochastic = false,
                    neighbourstateweight = stochastic ? .05 : 0.,
                    chosenactionweight = 2. /3.,
                    nbreak = 2, nadd = 4, changeprobability = 0.01)
    dm = DiscreteMaze(maze = maze, ngoals = ngoals, goalrewards=goalrewards,
                        stochastic = stochastic,
                        neighbourstateweight = neighbourstateweight,
                        chosenactionweight = chosenactionweight)
    switchflag = fill(false, 4, length(dm.maze))
    RandomChangeDiscreteMaze(dm, nbreak, nadd, changeprobability, switchflag,
                            seed, rng, chosenactionweight)
end
RandomChangeDiscreteMaze(maze; kwargs...) = RandomChangeDiscreteMaze(; maze = maze, kwargs...)
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
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function plotenv(env::DiscreteMaze)
    goals = env.goals
    # px = 6
    px = 16
    m = zeros(px*env.maze.dimx, px*env.maze.dimy)
    for i in 1:env.maze.dimx
        for j in 1:env.maze.dimy
            s = (i - 1) * env.maze.dimy + j
            iswall(env.maze, s, :down) && (m[(i-1)*px+1:i*px, j*px] .= -3)
            iswall(env.maze, s, :up) && (m[(i-1)*px+1:i*px, (j-1)*px+1] .= -3)
            iswall(env.maze, s, :right) && (m[i*px, (j-1)*px+1:j*px] .= -3)
            iswall(env.maze, s, :left) && (m[(i-1)*px+1, (j-1)*px+1:j*px] .= -3)
            if s == env.mdp.state
                val = -1
            elseif s in goals
                val = 1
            else
                continue
            end
            m[(i-1)*px+2:i*px-1, (j-1)*px+2:j*px-1] .= val
        end
    end
    imshow(m, clim = (-3, 3), colormap = 42)
end
