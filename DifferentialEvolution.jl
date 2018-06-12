#=
Simple version of differential evolution
Based off of https://github.com/rened/InformedDifferentialEvolution.jl
=#

__precompile__()

module DifferentialEvolution

export de, HIGH_VERBOSITY, LOW_VERBOSITY

# this should really take in an arbitrary axis
function matToCols(mat::Array{T, 2})::Array{Vector{T}, 1} where {T}
    return [mat[:, i] for i in 1:size(mat, 2)]
end

function vecVar(vecs::Array{Vector{Float64}, 1})::Float64
    mat = hcat(vecs...)
    return var(norm.(mat .- mean(mat, 2)))
end

HIGH_VERBOSITY = [:newline, :iter, :bestcost, :bestvec, :showvar]
LOW_VERBOSITY = [:alwaysiter, :bestcost, :showvar]

"""
Simple implementation of differential evolution optimization procedure
Given a cost function and relevant parameters, attemts to minimize value of 
supplied cost function

Parameters:
 - `costf` function to minimize
 - `mi` minimum input vector
 - `ma` maximal input vector
 - `nruns` number of iterations to run
 - `diffweight` multiplication constant on vector difference
 - `initpop` initial population of vectors
 - `io` output stream
 - `inrange` applies function to bring all vectors into desired range
 - `verbosity` options regarding logging
    - `:iter` shows iteration number 
    - `:newline` newline before iteration number
    - `:alwaysiter` shows iteration along with the rest of the parameters
    - `:bestcost` shows best cost at each run
    - `:bestvec` shows best vec at each run
    - `:pop` shows population at each run
    - `:showvar` shows cost variance at each run
"""
function de(costf::Function, mi::Vector, ma::Vector;
    npop = 100,
    nruns::Int = 1000,
    continueabove = Inf,
    diffweight::Float64 = 0.85,
    initpop = matToCols(mi .+ (rand(length(mi), npop) .* (ma - mi))),
    verbosity::Array = Symbol[], # 
    io = STDOUT,
    inrange = x -> x)

    pop = copy(initpop)
    pop = inrange.(pop)
    costs = costf.(pop)

    iter = 1

    bestcost, bestind = findmin(costs)
    variance = vecVar(pop)

    v_newline = in(:newline, verbosity)
    v_iter = in(:iter, verbosity)
    v_alwaysiter = in(:alwaysiter, verbosity)
    v_bestcost = in(:bestcost, verbosity)
    v_bestvec = in(:bestvec, verbosity)
    v_pop = in(:pop, verbosity)
    v_showvar = in(:showvar, verbosity)

    log(a...) = println(io, a...)

    function makenew(v1::Vector, c1::Float64, v2::Vector, v3::Vector)::Tuple{Vector, Float64}
        vn = v1 + (v2 - v3) * diffweight
        vn = inrange.(vn)
        cn = costf(vn)
        return cn < c1 ? (vn, cn) : (v1, c1)
    end 

    while iter <= nruns
        istr = v_alwaysiter ? " #$(iter)" : ""
        varstr = v_showvar ? " var = $(variance)" : ""
        v_newline && log("")
        v_iter && log("Iteration #$(iter)")
        v_bestcost && log("Best cost$(istr): $(bestcost) $(varstr)")
        v_bestvec && log("Best vec$(istr): $(pop[bestind])")
        v_pop && log("Population$(istr): $(pop)")

        iter += 1
        newpop = Vector{Float64}[]
        newcosts = Float64[]
        for i = 1:npop
            r1 = rand(1:npop)
            r2 = rand(1:npop)
            r3 = rand(1:npop)

            v, c = makenew(pop[r1], costs[r1], pop[r2], pop[r3])
            push!(newpop, v)
            push!(newcosts, c)
        end

        pop = newpop
        costs = newcosts
        bestcost, bestind = findmin(costs)
        variance = vecVar(pop)
    end

    return pop[bestind], bestcost
end

end # module

function demo()
    f(x) = sum(abs.(x))
    mi = [-1,-1]; ma = [1,1]
    de(f, mi, ma, maxiter = 1000, verbosity=HEAVY_VERBOSITY) 
end