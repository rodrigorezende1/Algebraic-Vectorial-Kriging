"""
SPDX-FileCopyrightText: © 2019 Rodrigo Silva Rezende <rodrigo.silvarezende@tu-berlin.de>

SPDX-License-Identifier: BSD-3-Clause

This Module contains all the functions necessary to build the kriging model
of a give input/ouput relation i.e. objectiv function and it contains the kriging
function itself that uses all these functions to perform the construction of the
final surrogate.
This program is a modified implementation of the Kriging approach presented in the
Engineering Design via Surrogate Modelling - A.I.J. Forrester et. al. with some
modifications.
Kriging is a sucessor of Kriging_first, implementing the same things as in the
first but also capable of receiving more than one variable to train the Kriging
Model.

This code was developed by: Rodrigo Silva Rezende during his master thesis at
the TU Berlin in the Theoretische Elektrotechnik Department - 25/11/2019

Update 1: MLE, genetic algorithm, Nelder Mead and performance - 25/01/2020
"""



module Kriging

    using LinearAlgebra #Dealing with matricies
    using Optim #MLE Nelder Mead
    using BlackBoxOptim #MLE Genetic
    using SpecialFunctions #Infill Criteria
    export kriging, cormatrix, vectcor, mle, hyperpargenetic, hyperparNelderMead,hyperparmix, PredMSEgenetic, PredMSENelderMead, ExpImpgenetic, ExpImpNelderMead

    """
        kriging(xₑ::Vector{Float64}, Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64},θ::Vector{Float64},p::Vector{Float64})

    Generates the kriging model for the given model.

    ### Input

    - `xₑ` -- vector of evaluation points in which the given model should be tested
    - `Xₜ` -- matrix cointaining the training input base data of the given model
    - `Yₜ` -- matrix cointaining the training output training base data of the give
              model
    - `θ` -- Hyperparameter 1 of the Kriging model with dimension [Nᵥ x 1],
             with Nᵥ being the number of different variables of the model
    - `p` -- Hyperparameter 2 of the Kriging model with dimension 1 (Skalar)

    ### Output

    Matrix containing the Kriging model of the given input/output relation.

    ### Notes

    The kriging function implemented here calls all the building blocks as extern
    functions. At the end of the function the kriging model Rₛₖᵣ is created as a
    simply matrix multiplication.

    ### Algorithm

    Generates the kriging model with the training output data Yₜ with the
    training input data Xₜ and evaluate this model at the point xₑ. For that, the
    kriging(xₑ, Xₜ, Yₜ) calls the cormatrix(Xₜ) function to construct the
    correlation matrix of the Xₜ data and the vectcor(xₑ, Xₜ) to form the vector
    of correlations between the evaluated point xₑ and the Xₜ.

    """
    function kriging(xₑ::Vector{Float64}, Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64},θ::Vector{Float64},p::Vector{Float64}) #Xₜ is the base training set and Yₜ is the associated fine model response of it; Xₑ is the scenario to be evaluated
          One = ones(size(Xₜ,1),1)
          ψₘₐₜ = cormatrix(Xₜ,θ,p)
          U = cholesky(ψₘₐₜ).U
          μₒₚₜ = ((One)'*(U\(U'\Yₜ)))./((One)'*(U\(U'\One))) #The pointwise operation here is due to the vector output possibility
          ψₑᵥₐₗ = vectcor(xₑ, Xₜ,θ,p) # Vector of correlations between the point Xₑ and the base training Xₜ
          Yₑᵥₐₗ = (μₒₚₜ+ψₑᵥₐₗ'*(U\(U'\(Yₜ-One*μₒₚₜ))))'[:,:]
          return Yₑᵥₐₗ
    end

    """
        cormatrix(Xₜ::Matrix{Float64},θ::Vector{Float64},p::Vector{Float64})

    Calculates the correlation matrix of the data Xₜ with a exponential correlation
    function.

    ### Input

    - `Xₜ` -- matrix with the values that will generate each element of the
              correlation matrix
    - `θ` -- Hyperparameter 1 of the Kriging model with dimension [Nᵥ x 1],
                       with Nᵥ being the number of different variables of the model
    - `p` -- Hyperparameter 2 of the Kriging model with dimension 1 (Skalar)


    ### Output

    Correlation matrix ψₘₐₜ of the elements of Xₜ.

    ### Algorithm

    The cormatrix(Xₜ::) function implements the construction of the correlation
    matrix and for that it uses the correlation function ψ(x₁, x₂) that uses a
    exponential relationship between the points x₁ and x₂.
    """
    function cormatrix(Xₜ::Matrix{Float64},θ::Vector{Float64},p::Vector{Float64})
        ψₘₐₜ = zeros(size(Xₜ, 1), size(Xₜ, 1))
        @inbounds for i in 1:size(Xₜ, 1)
            for j in i+1:size(Xₜ, 1) #Constructing only the upper half
                ψₘₐₜ[i, j] = exp(-sum(θ.*(abs.(Xₜ[i, :]-Xₜ[j, :])).^p)) #Function abs() is really important here
            end
        end
        ψₘₐₜ = ψₘₐₜ + ψₘₐₜ'+I+I*1e-200 #Constructing ψₘₐₜ and adding a small number to reduce ill conditioning
        return ψₘₐₜ
    end

    """
        vectcor(xₑ::Union{Matrix{Float64},Number}, Xₜ::Matrix{Float64},θ::Union{Matrix{Float64},Number},p::Number)

    Compute the correlation vector of two vectors using an exponential correlation
    function

    ### Input

    - `xₑ` -- first vector of the calculation
    - `Xₜ` -- matrix or second vector of the calculation
    - `θ` -- Hyperparameter 1 of the Kriging model with dimension [Nᵥ x 1],
                       with Nᵥ being the number of different variables of the model
    - `p` -- Hyperparameter 2 of the Kriging model with dimension 1 (Skalar)

    ### Output

    A vector with elements representing the correlation between the elements of xₑ
    and Xₜ.

    ### Algorithm

    For the calculation of the correlation between each element of of xₑ
    and Xₜ the vectcor(xₑ, Xₜ) uses the correlation function  ψ(x₁, x₂) that
    uses a exponential relationship between the points x₁ and x₂.
    """
    function vectcor(xₑ::Vector{Float64}, Xₜ::Matrix{Float64},θ::Vector{Float64},p::Vector{Float64}) #Here xₑ is every element of Xₑ. So this calculation is done for every evaluation point
        ψₑᵥₐₗ = ones(size(Xₜ, 1),1)
        @inbounds for i in 1:size(Xₜ, 1)
            ψₑᵥₐₗ[i,1] = exp(-sum(θ.*(abs.(xₑ.-Xₜ[i, :])).^p)) #The .- is necessary since Xₜ[i, :]::Array{Float64,1}
        end
        return ψₑᵥₐₗ
    end

    """
        mle(hyperpar::Union{Union{Matrix{Float64},Array{<:Number,1}},Number})

    Calculates the Maximum Likelihood Estimation required by the Kriging Process
    with the given parameters and for a given Data set Yₜᵍ(Output) and Xₜᵍ (Input)
    of the objective function. The data set is defined as global by the caller
    of the mle()

    ### Input

    - `hyperpar` --  Hyperparameter 1 - "θ" and Hyperparameter 2 - "p" of the Kriging
                     model (both variables are concatenated inside hyperpar),
                     such that hyperpar = [θ, p]

    ### Output

    The Maximum Likelihood Estimation

    ### Algorithm

    The Maximum Likelihood Estimation implements the algorithm discussed in
    in the Engineering Design via Surrogate Modelling - A.I.J. Forrester et. al.
    """
    function mle(hyperpar::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
        hyperpar = hyperpar[:]
        #I will have to change here
        p = hyperpar[end:end]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
        θ = 10 .^hyperpar[1:end-1] #Searching in a log scale
        Nₖᵣ = size(Xₜᵍ, 1)
        Nₒᵤₜ = size(Yₜᵍ,2)
        One = ones(Nₖᵣ, 1)
        ψₘₐₜ = cormatrix(Xₜᵍ, θ, p)
        if isposdef(ψₘₐₜ)
            U = cholesky(ψₘₐₜ).U
            LnDetψ = 2*sum(log.(abs.(diag(U))))
            μₘₗₑ = ((One)'*(U\(U'\Yₜᵍ)))./((One)'*(U\(U'\One)))
            σ² = ((Yₜᵍ-One*μₘₗₑ)'*(U\(U'\(Yₜᵍ-One*μₘₗₑ))))/Nₖᵣ
            σ² = diag(σ²)
            LnL = sum(-1 .*(-(Nₖᵣ/2).*log.(σ²).-LnDetψ/2))/Nₒᵤₜ #Nₒᵤₜ f(x) per scenario
        else #Penalty for ill-conditioned
                LnL = 10000.0
            end
        return LnL
    end

    """
        hyperpargenetic(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64})

    Calculates a local best set of values of "θ" and "p" required to build
    the Kriging Model of an objective function. For that the hyperpargenetic uses
    the Maximum Likelihood Estimation of Xₜ and Yₜ using values of "θ" and "p"
    set by an genetic algorithm. The MLE is calculated with the function mle().

    ### Input

    - `Xₜ` -- matrix cointaining the training input base data of the given model
    - `Yₜ` -- matrix cointaining the training output training base data of the give
              model

    ### Output

    hyperpargenetic returns as output: the hyperpar ("θ" and "p" concatenated),
    θ and p themselves.

    ### Algorithm

    hyperpargenetic implements the algorithm discussed in the Engineering Design
    via Surrogate Modelling - A.I.J. Forrester et. al.
    The genetic algorithm used is the one offered by the package Optim.jl
    """
    function hyperpargenetic(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64})
            global Xₜᵍ, Yₜᵍ
            Xₜᵍ = Xₜ
            Yₜᵍ = Yₜ
            Nᵥ = size(Xₜᵍ, 2)
            res = bboptimize(mle; SearchRange = (-3.0, 2.0), NumDimensions = (Nᵥ+1),   Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 300.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange its really important! #Searching [0.5999,2.0]
            hyperpar = best_candidate(res)
            hyperpar = reshape(hyperpar, size(hyperpar,1),1)
            θ = 10 .^hyperpar[1:end-1][:]
            p = hyperpar[end:end, 1]./5 .+1.599
            return θ, p
    end

    """
        hyperparNelderMead(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64})

    Calculates a local best set of values of "θ" and "p" required to build
    the Kriging Model of an objective function. For that the hyperparNelderMead
    uses the Maximum Likelihood Estimation of Xₜ and Yₜ using values of "θ" and "p"
    set by the Nelder Mead algorithm. The MLE is calculated with the function mle().

    ### Input

    - `Xₜ` -- matrix cointaining the training input base data of the given model
    - `Yₜ` -- matrix cointaining the training output training base data of the give
              model

    ### Output

    hyperparNelderMead returns as output: the hyperpar ("θ" and "p" concatenated),
    θ and p themselves.

    ### Algorithm

    hyperparNelderMead implements the algorithm discussed in the Engineering Design
    via Surrogate Modelling - A.I.J. Forrester et. al. but instead of using an
    genetic alrogithm to search for the best values of "θ" and "p", it uses the
    Nelder Mead algorithm
    The Nelder Mear algorithm used is the one offered by the package using BlackBoxOptim.jl
    More about the Nelder Mead algorithm:
    https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    """
    function hyperparNelderMead(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64})
            global Xₜᵍ, Yₜᵍ
            Xₜᵍ = Xₜ
            Yₜᵍ = Yₜ
            Nᵥ = size(Xₜᵍ, 2)
            lower = ones((Nᵥ+1),1)*-3.0
            upper = ones((Nᵥ+1),1)*2.0
            initial_x = ones((Nᵥ+1),1)*-1
            result = optimize(mle, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 300.0)) #time_limit = 10 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            hyperpar = Optim.minimizer(result)
            hyperpar = reshape(hyperpar, size(hyperpar,1),1)
            θ = 10 .^hyperpar[1:end-1][:]
            p = hyperpar[end:end, 1]./5 .+1.599
            return θ, p
    end

    """
        hyperparmix(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64})

    Calculates a local best set of values of "θ" and "p" required to build
    the Kriging Model of an objective function. For that the hyperpar uses
    the Maximum Likelihood Estimation of Xₜ and Yₜ using values of "θ" and "p"
    set by an mixed genetic and Nelder Mead algorithm. The MLE is calculated
    with the function mle().

    ### Input

    - `Xₜ` -- matrix cointaining the training input base data of the given model
    - `Yₜ` -- matrix cointaining the training output training base data of the give
              model

    ### Output

    hyperpargenetic returns as output: the hyperpar ("θ" and "p" concatenated),
    θ and p themselves.

    ### Algorithm

    hyperpargenetic implements the algorithm discussed in the Engineering Design
    via Surrogate Modelling - A.I.J. Forrester et. al.
    The genetic algorithm used is the one offered by the package Optim.jl
    """
    function hyperparmix(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64})
            global Xₜᵍ, Yₜᵍ
            Xₜᵍ = Xₜ
            Yₜᵍ = Yₜ
            Nᵥ = size(Xₜᵍ, 2)
            res = bboptimize(mle; SearchRange = (-3.0, 2.0), NumDimensions = (Nᵥ+1),   Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 300.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange its really important! #Searching [0.5999,2.0]
            hyperpargen = best_candidate(res)
            hyperpargen = reshape(hyperpargen, size(hyperpargen,1),1)
            lower = ones((Nᵥ+1),1)*-3.0
            upper = ones((Nᵥ+1),1)*2.0
            result = optimize(mle, lower, upper, hyperpargen, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 300.0)) #time_limit = 10 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            hyperpar = Optim.minimizer(result)
            hyperpar = reshape(hyperpar, size(hyperpar,1),1)
            θ = 10 .^hyperpar[1:(size(hyperpar,1)-1)][:]
            p = hyperpar[end:end, 1]./5 .+1.599
            return θ, p
    end


###################### Infill Criteria ############################

    ### Implementation of the s²(x)-based infill criteria

    function PredMSE(xₑ::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
        xₑ =  reshape(xₑ, 1,length(xₑ))
        xₑ = round.(xₑ, digits=3)
        Xₜ_comp = round.(Xₜᵍ, digits=3)
        if issubset(xₑ,Xₜ_comp) #Points nearby up to 4 digits will be penalized
            s² = 100.0
        else
            One = ones(size(Xₜᵍ,1),1)
            Nₖᵣ = size(Yₜᵍ,1)
            Nₒᵤₜ = size(Yₜᵍ,2)
            ψₑᵥₐₗ = vectcor(xₑ, Xₜᵍ,θᵍ,pᵍ) # Vector of correlations between the point Xₑ and the base training Xₜ
            σ² = ((Yₜᵍ-One*μₒₚₜᵍ)'*(Uᵍ\(Uᵍ'\(Yₜᵍ-One*μₒₚₜᵍ))))/Nₖᵣ
            σ² = diag(σ²)
            s² = sum(σ² * (1-(ψₑᵥₐₗ'*(Uᵍ\Uᵍ'\ψₑᵥₐₗ))[1,1]))/Nₒᵤₜ
            s² = -s²
        end
        return s²
    end

    function PredMSEgenetic(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64},θ::Union{Union{Matrix{Float64},Array{<:Number,1}},Number},p::Number)
        global Xₜᵍ, Yₜᵍ,ψₘₐₜᵍ, Uᵍ, μₒₚₜᵍ,θᵍ,pᵍ
        Xₜᵍ = Xₜ
        Yₜᵍ = Yₜ
        θᵍ = θ
        pᵍ = p
        One = ones(size(Xₜ,1),1)
        Nᵥ = size(Xₜ,2)
        Nₖᵣ = size(Yₜ,1)
        Nₒᵤₜ = size(Yₜ,2)
        ψₘₐₜ = cormatrix(Xₜ,θ,p)
        ψₘₐₜᵍ = ψₘₐₜ
        U = cholesky(ψₘₐₜᵍ).U
        Uᵍ = U
        μₒₚₜᵍ = ((One)'*(U\(U'\Yₜ)))./((One)'*(U\(U'\One))) #The pointwise operation here is due to the vector output possibility
        res = bboptimize(PredMSE; SearchRange = (0.0, 1.0), NumDimensions = (Nᵥ),   Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 300.0, PopulationSize = 50)
        xₑ_best = best_candidate(res)
        xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
        return xₑ_best
    end

    function PredMSENelderMead(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64},θ::Union{Union{Matrix{Float64},Array{<:Number,1}},Number},p::Number)
        global Xₜᵍ, Yₜᵍ,ψₘₐₜᵍ, Uᵍ, μₒₚₜᵍ,θᵍ,pᵍ
        Xₜᵍ = Xₜ
        Yₜᵍ = Yₜ
        θᵍ = θ
        pᵍ = p
        One = ones(size(Xₜ,1),1)
        Nᵥ = size(Xₜ,2)
        Nₖᵣ = size(Yₜ,1)
        Nₒᵤₜ = size(Yₜ,2)
        ψₘₐₜ = cormatrix(Xₜ,θ,p)
        ψₘₐₜᵍ = ψₘₐₜ
        U = cholesky(ψₘₐₜᵍ).U
        Uᵍ = U
        μₒₚₜᵍ = ((One)'*(U\(U'\Yₜ)))./((One)'*(U\(U'\One))) #The pointwise operation here is due to the vector output possibility
        lower = zeros(Nᵥ,1)
        upper = ones(Nᵥ,1)
        initial_x = ones(Nᵥ,1)*0.5
        start_time = time()
        result = optimize(PredMSE, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 300.0)) #time_limit = 100 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
        xₑ_best = Optim.minimizer(result)
        xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
        return xₑ_best
    end

    ### Implementation of the Expectation of Impromevement-based infill criteria
    function ExpImp(xₑ::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
        xₑ =  reshape(xₑ, 1,length(xₑ))
        xₑ = round.(xₑ, digits=3)
        Xₜ_comp = round.(Xₜᵍ, digits=3)
        if issubset(xₑ,Xₜ_comp) #Points nearby up to 3 digits will be penalized
            ExImp = 100.0

        else
            One = ones(size(Xₜᵍ,1),1)
            Nₖᵣ = size(Yₜᵍ,1)
            Nₒᵤₜ = size(Yₜᵍ,2)
            ψₑᵥₐₗ = vectcor(xₑ, Xₜᵍ,θᵍ,pᵍ) # Vector of correlations between the point Xₑ and the base training Xₜ
            σ² = ((Yₜᵍ-One*μₒₚₜᵍ)'*(Uᵍ\(Uᵍ'\(Yₜᵍ-One*μₒₚₜᵍ))))/Nₖᵣ
            σ² = diag(σ²)
            Yₑᵥₐₗ = (μₒₚₜᵍ+ψₑᵥₐₗ'*(Uᵍ\(Uᵍ'\(Yₜᵍ-One*μₒₚₜᵍ))))
            Yₑᵥₐₗ = convert(Array,Yₑᵥₐₗ)
            #s² = sum(σ² * (1-ψₑᵥₐₗ'*(U\U'\ψₑᵥₐₗ)))/Nₒᵤₜ #Vectorial Kriging? sum(σ² * (1-ψₑᵥₐₗ'*(U\U'\ψₑᵥₐₗ)))/Nₖᵣ
            s² = (σ² * (1-(ψₑᵥₐₗ'*(Uᵍ\Uᵍ'\ψₑᵥₐₗ))[1,1]))[:,:]'
            s = (sqrt.(abs.(s²)))
            yₘᵢₙ = Yₜᵍ[argmin(sum(Yₜᵍ, dims=2))[1],:]' #Summing all the elements of each row and deciding which row has the lowest sum of elements

            #Expected Improvement
            if sum(s²) == 0
                ExImp = 100.0
            elseif minimum((1/sqrt(2))*(yₘᵢₙ-Yₑᵥₐₗ)./s) < -5
                xterm1 = ((1/sqrt(2))*(yₘᵢₙ-Yₑᵥₐₗ)./s)
                term = zeros(10,Nₒᵤₜ)
                for i=1:10
                    term[i,:] = ((-1)^(i-1)*(factorial(2*(i-1))/(2^(i-1)*factorial(i-1)))./(2^(i-1)))*xterm1.^(-(2*(i-1)+1))
                end
                B = (yₘᵢₙ-Yₑᵥₐₗ)*(1/(2*sqrt(pi)))*sum(term, dims=1)+(s/sqrt(2*π))
                if minimum(B) > 0
                    ExImp = (log10(2)/log(2))*(log.(B)-(1/2)*((yₘᵢₙ-Yₑᵥₐₗ).^2 ./s²)) #abs(B)?
                    ExImp = -sum(ExImp)
                else
                    ExImp = 100.0
                end
            else
                ExImp1 = (yₘᵢₙ-Yₑᵥₐₗ).*(0.5 .+0.5*erf.((yₘᵢₙ-Yₑᵥₐₗ)./ (sqrt.(abs.(s²))*sqrt(2)))) #Vectorial Kriging?
                ExImp2 = s*(1/sqrt(2*π)).*exp.(-(yₘᵢₙ-Yₑᵥₐₗ).^2 ./(2*s²))
                if minimum(ExImp1 + ExImp2) > 0
                ExImp = -sum(log10.(ExImp1 + ExImp2))
                else #minimizer
                ExImp = 100.0
                end
            end
        end
        return ExImp
    end

    function ExpImpgenetic(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64},θ::Union{Union{Matrix{Float64},Array{<:Number,1}},Number},p::Number)
        global Xₜᵍ, Yₜᵍ,ψₘₐₜᵍ, Uᵍ, μₒₚₜᵍ,θᵍ,pᵍ
        Xₜᵍ = Xₜ
        Yₜᵍ = Yₜ
        θᵍ = θ
        pᵍ = p
        One = ones(size(Xₜ,1),1)
        Nᵥ = size(Xₜ,2)
        Nₖᵣ = size(Yₜ,1)
        Nₒᵤₜ = size(Yₜ,2)
        ψₘₐₜ = cormatrix(Xₜ,θ,p)
        ψₘₐₜᵍ = ψₘₐₜ
        U = cholesky(ψₘₐₜᵍ).U
        Uᵍ = U
        μₒₚₜᵍ = ((One)'*(U\(U'\Yₜ)))./((One)'*(U\(U'\One))) #The pointwise operation here is due to the vector output possibility
        res = bboptimize(ExpImp; SearchRange = (0.0, 1.0), NumDimensions = (Nᵥ),   Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 300.0, PopulationSize = 50)
        xₑ_best = best_candidate(res)
        xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
        return xₑ_best
    end

    function ExpImpNelderMead(Xₜ::Matrix{Float64}, Yₜ::Matrix{Float64},θ::Union{Union{Matrix{Float64},Array{<:Number,1}},Number},p::Number)
        global Xₜᵍ, Yₜᵍ,ψₘₐₜᵍ, Uᵍ, μₒₚₜᵍ,θᵍ,pᵍ
        Xₜᵍ = Xₜ
        Yₜᵍ = Yₜ
        θᵍ = θ
        pᵍ = p
        One = ones(size(Xₜ,1),1)
        Nᵥ = size(Xₜ,2)
        Nₖᵣ = size(Yₜ,1)
        Nₒᵤₜ = size(Yₜ,2)
        ψₘₐₜ = cormatrix(Xₜ,θ,p)
        ψₘₐₜᵍ = ψₘₐₜ
        U = cholesky(ψₘₐₜᵍ).U
        Uᵍ = U
        μₒₚₜᵍ = ((One)'*(U\(U'\Yₜ)))./((One)'*(U\(U'\One))) #The pointwise operation here is due to the vector output possibility
        lower = zeros(Nᵥ,1)
        upper = ones(Nᵥ,1)
        initial_x = ones(Nᵥ,1)*0.5
        start_time = time()
        result = optimize(ExpImp, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 300.0)) #time_limit = 100 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
        xₑ_best = Optim.minimizer(result)
        xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
        return xₑ_best
    end

end
