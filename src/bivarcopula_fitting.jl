using ForwardDiff, Ipopt, Copulas, Distributions, StableRNGs

include("ipopt_interface.jl")

function pdf_Rotated_BivariateCopula(u₁,u₂, p, copulatype)
    θ₁ = p[1]; θ₂ = p[2]; θ₃ = p[3]; θ₄ = p[4]; w₁ = p[5]; w₂ = p[6]; w₃ = p[7]; w₄ = 1.0 - w₁ - w₂ - w₃

    if copulatype == :gumbel
        C = GumbelCopula
    elseif copulatype == :clayton
        C = ClaytonCopula
    elseif copulatype == :joe
        C = JoeCopula
    end

    oneminusu₁ = 1.0 .- u₁
    oneminusu₂ = 1.0 .- u₂

    return w₁*bivarpdf(C(2,θ₁),u₁,u₂) + w₂*bivarpdf(C(2,θ₂),oneminusu₁,u₂) + w₃*bivarpdf(C(2,θ₃),oneminusu₁,oneminusu₂) + w₄*bivarpdf(C(2,θ₄),u₁,oneminusu₂)
end


function cdf_Rotated_BivariateCopula(u₁,u₂, p, copulatype)
    θ₁ = p[1]; θ₂ = p[2]; θ₃ = p[3]; θ₄ = p[4]; w₁ = p[5]; w₂ = p[6]; w₃ = p[7]; w₄ = 1.0 - w₁ - w₂ - w₃

    if copulatype == :gumbel
        C = GumbelCopula
    elseif copulatype == :clayton
        C = ClaytonCopula
    elseif copulatype == :joe
        C = JoeCopula
    end

    oneminusu₁ = 1.0 .- u₁
    oneminusu₂ = 1.0 .- u₂

    return w₁*bivarcdf(C(2,θ₁),u₁,u₂) + w₂*bivarcdf(C(2,θ₂),oneminusu₁,u₂) + w₃*bivarcdf(C(2,θ₃),oneminusu₁,oneminusu₂) + w₄*bivarcdf(C(2,θ₄),u₁,oneminusu₂)
end

@inline weightconstr(p) = sum(p[5:7])

function fit_Rotated_BivariateCopula(u₁::AbstractVector, u₂::AbstractVector, 
    init, copulatype;
    tol=1.0e-5, maxit=500, print_level=5, Rsafe = false, censored = false)

    # Box constraints (lowerb and upperb) and inequality constraints for the way
    # that I've implemented the nonlinear constraints.
    lowerb_weights = repeat([0.0], 3)
    upperb_weights = repeat([1.0], 3)

    if copulatype == :gumbel
        lowerb_parms = repeat([1.0],4)
        upperb_parms = repeat([100.0],4)
        C = GumbelCopula
    elseif copulatype == :clayton
        lowerb_parms = repeat([0.0],4) #repeat([-1.0],4)
        upperb_parms = repeat([100.0],4)
        C = ClaytonCopula
    elseif copulatype == :joe
        lowerb_parms = repeat([1.0],4)
        upperb_parms = repeat([100.0],4)
        C = JoeCopula
    end

    obj = function(p)
        θ₁ = p[1]; θ₂ = p[2]; θ₃ = p[3]; θ₄ = p[4]; w₁ = p[5]; w₂ = p[6]; w₃ = p[7]; w₄ = 1.0 - w₁ - w₂ - w₃
        ((θ₁ ≤ lowerb_parms[1]) || (θ₂ ≤ lowerb_parms[2]) || (θ₃ ≤ lowerb_parms[3]) || (θ₄ ≤ lowerb_parms[4]) || (w₁ ≤ 0.0) || (w₂ ≤ 0.0) || (w₃ ≤ 0.0) || (w₄ ≤ 0.0)) && return Inf
        mat = pdf_Rotated_BivariateCopula(u₁,u₂,p,copulatype)
        (minimum(mat) ≤ 0.0) && return Inf
        return -sum(log,mat)
    end
    
    lowerb = [lowerb_parms; lowerb_weights]
    upperb = [upperb_parms; upperb_weights]

    cons_low = [0.0]
    cons_upp = [1.0]
    
    grad!(p,g) = ForwardDiff.gradient!(g, obj, p)
    hess(p)    = ForwardDiff.hessian(obj, p)

    constrf     = (weightconstr)
    constr(p)   = [constrf(p)] # (iii)

    # Write the constraint in the final format that ipopt actually wants.
    ipopt_constr(p,g) = (g .= constr(p))

    # A properly reshaped jacobian (not very pretty sorry).
    g_jac(p) = vec(Matrix(ForwardDiff.jacobian(constr, p)'))

    # Prepare the final hessian and constraint jacobian functions.
    nconstr  = 1
    constrfh = [p->ForwardDiff.hessian(constrf, p)]
    ipopt_hess = (x,r,c,o,l,v) -> ipopt_hessian(x,r,c,o,l,v,hess,constrfh,nconstr)
    ipopt_jac_constr = (x,r,c,v) -> ipopt_constr_jac(x,r,c,v,g_jac,nconstr)

    # Set up the problem:
    prob = CreateIpoptProblem(
                            length(init), # number of parameters
                            lowerb, upperb, # box for parameter constraints
                            1, # number of nonlinear constraints
                            cons_low, cons_upp, # box for nonlinear constraint fxns
                            length(init), # size of CONSTRAINT JACOBIAN
                            div(length(init)*(length(init)+1), 2), # size of ltri of HESSIAN
                            obj,   
                            ipopt_constr,
                            grad!, 
                            ipopt_jac_constr,
                            ipopt_hess)
    AddIpoptStrOption(prob, "sb", "yes")
    AddIpoptNumOption(prob, "tol", tol)
    AddIpoptIntOption(prob, "max_iter", maxit)
    AddIpoptIntOption(prob, "print_level", Int(print_level))
    prob.x = deepcopy(init)
    status = IpoptSolve(prob)

    return (status = status, mle = (θ₁ = prob.x[1], θ₂ = prob.x[2], θ₃ = prob.x[3], θ₄ = prob.x[4], w₁ = prob.x[5], w₂ = prob.x[6], w₃ = prob.x[7], w₄ = 1.0 - sum(prob.x[5:7])), nll = obj(prob.x))
end

function fit_Rotated_BivariateCopula_robust(u₁,u₂, inits, copulatype; num_attempts = 10, kwargs...)
    stable_rng = StableRNG(11235) # seed here.
    initlen    = length(first(inits))

    initthetas = first(inits)[1:4]

    try
        for j in 1:max(Int(num_attempts), length(inits))
            if (j <= length(inits)) 
                initj =  inits[j]
            else rand(stable_rng, initlen)
                ws = rand(stable_rng,Dirichlet(4,1))[1:3]
                initj = [initthetas; ws]
            end
            mle_attempt = fit_Rotated_BivariateCopula(u₁,u₂, initj, copulatype; kwargs...)
            iszero(mle_attempt[1]) && return mle_attempt
            @info "Optimization failed for initialization $j. Moving on..."
        end
        @warn "No optimization attempt successful."
        return (-1, zeros(initlen).*NaN, zeros(initlen,initlen).*NaN)
    catch er
        @warn "Optimization error: $er"
    end
end

