"""
This Module contains all the functions necessary to build the kriging model
of a give input/ouput relation i.e. evaluated model and it contains the kriging
function itself that uses all these function to perform the construction of the
model. This program is an implementation of the Kriging approach present in
Variable-Fidelity Electromagnetics and Co-Kriging for Accurate Modeling of
Antennas - Koziel et.al.

This version of Kriging is deprecated - 29/01/2020
"""


module Kriging

    using LinearAlgebra #Dealing with matricies
    export kriging, ψ, cormatrix, vectcor

    """
        kriging(xₑ::Union{Union{Array{Float64,2},Array{Int64,2}},Number},
        Xₜ::Union{Array{Float64,2},Array{Int64,2}}, Rₕ::Union{Array{Float64,2},
        Array{Int64,2}})

    Generates the kriging model for the given model.

    ### Input

    - `xₑ` -- vector of evaluation points in which the given model should be tested
    - `Xₜ` -- matrix cointaining the training input base data of the given model
    - `Rₕ` -- matrix cointaining the training output training base data of the give
              model

    ### Output

    Matrix containing the Kriging model of the given input/output relation.

    ### Notes

    The kriging function implemented here calls all the building blocks as extern
    functions. At the end of the function the kriging model Rₛₖᵣ is created as a
    simply matrix multiplication.

    ### Algorithm

    Generates the kriging model with the training output data Rₕ with the
    training input data Xₜ and evaluate this model at the point xₑ. For that, the
    kriging(xₑ, Xₜ, Rₕ) calls the functions ψ(x₁, x₂) to build the correlation
    function; the cormatrix(Xₜ) to construct the correlation matrix for the Xₜ data
    ; and the vectcor(xₑ, Xₜ) to form the vector of correlations between the
    evaluated point xₑ and the Xₜ.

    """
    function kriging(xₑ::Union{Union{Array{Float64,2},Array{Int64,2}},Number}, Xₜ::Union{Array{Float64,2},Array{Int64,2}}, Rₕ::Union{Array{Float64,2},Array{Int64,2}}) #Xₜ is the base training set and Rₕ is the associated fine model response of it Xₑ is the element to be evaluated
        M = 1 #M is a value being constant
        F = ones(size(Rₕ, 1), 1) #F is one Vector with same first dimensions as Rₕ
        ψmat = cormatrix(Xₜ)
        α = inv((Xₜ)'*inv(ψmat)*Xₜ)*(Xₜ)'*inv(ψmat)*Rₕ #ψmat is the correlation matrix
        r = vectcor(xₑ, Xₜ) # Vector of correlations between the point Xₑ and the base training Xₜ
        Rₛₖᵣ = (M*α+r*inv(ψmat)*(Rₕ-F*α))'
        return Rₛₖᵣ
    end

    """
        ψ(x₁::Union{Float64,Int64}, x₂::Union{Float64,Int64})

    Calculates the the correlation function between the two inputs x₁,x₂.

    ### Input

        - `x₁` -- value of the first point
        - `x₂` -- value of the second point

    ### Output

    An Int64 value meaning the correlation between the values x₁ and x₂.

    ### Algorithm

    The correlation function implemented here is formed by an exponential in the
    form exp(-θ * abs(x₁-x₂)^(p)), where θ and p are constants and play a important
    and diferents roles for diferent problems and therefore for diferent set of
    input vectors.
    """
    function ψ(x₁::Union{Float64,Int64}, x₂::Union{Float64,Int64}) # This is the correlation function
        p = 1 # It can also be 2 if the problem is smooth
        θ = 0.01 # The first values will be guessed but it can be calculated with a MLE. The guesses are between 0.1 and 10
        corfunc = exp(-θ * abs(x₁-x₂)^(p))
        return corfunc
    end

    """
        cormatrix(Xₜ::Union{Array{Float64,2},Array{Int64,2}})

    Calculates the correlation matrix of the data Xₜ with a exponential correlation
    function.

    ### Input

    - `Xₜ` -- matrix with the values that will generate each element of the
              correlation matrix

    ### Output

    Correlation matrix of the elements of Xₜ.

    ### Algorithm

    The cormatrix(Xₜ::) function implements the construction of the correlation
    matrix and for that it uses the correlation function ψ(x₁, x₂) that uses a
    exponential relationship between the points x₁ and x₂.
    """
    function cormatrix(Xₜ::Union{Array{Float64,2},Array{Int64,2}})
        ψmat = ones(size(Xₜ, 1), size(Xₜ, 1))
        for i in 1:size(Xₜ, 1)
            for j in 1:size(Xₜ, 1)
                ψmat[i, j] = ψ(Xₜ[i, 1], Xₜ[j, 1])
            end
        end
        return ψmat
    end

    """
        vectcor(xₑ::Union{Union{Array{Float64,2}, Xₜ::Union{Array{Float64,2},Array{Int64,2}})

    Compute the correlation vector of two vectors using an exponential correlation
    function

    ### Input

    - `xₑ` -- first vector of the calculation
    - `Xₜ` -- matrix or second vector of the calculation

    ### Output

    A vector with elements representing the correlation between the elements of xₑ
    and Xₜ.

    ### Algorithm

    For the calculation of the correlation between each element of of xₑ
    and Xₜ the vectcor(xₑ::, Xₜ::) uses the correlation function  ψ(x₁, x₂) that
    uses a exponential relationship between the points x₁ and x₂.
    """
    function vectcor(xₑ::Union{Union{Array{Float64,2},Array{Int64,2}},Number}, Xₜ::Union{Array{Float64,2},Array{Int64,2}}) #Here xₑ is every element of Xₑ. So this calculation is done for every evaluation point
        r = ones(1, size(Xₜ, 1))
        for i in 1:size(Xₜ, 1)
            r[1, i] = ψ(xₑ, Xₜ[i, 1])
        end
        return r
    end

end
