"""
This Module contains the evaluation of the Eggholder function so that surrogate
models can be evaluated.

This code was developed by: Rodrigo Silva Rezende during his master thesis at
the TU Berlin in the Theoretische Elektrotechnik Department - 05/01/2020
"""

module Eggholder

    using PyPlot
    export Eggholder_func

    """
        Eggholder(x₁::Array{<:Number,2},x₂::Array{<:Number,2})

    Calculates the Eggholder function for the interval defined by x₁ and x₂.

    ### Input

        - `x₁` -- interval of the first variable in the Eggholder function
        - `x₂` -- interval of the second variable in the Eggholder function

    ### Output

    A Matrix with the values of the Eggholder function in each point

    ### Algorithm

    The output Matrix is a i X j matrix containing the evaluations of the
    Eggholder function on the points [x₁(i);x₂(j)]
    """
    function Eggholder_func(x₁::Array{<:Number,2},x₂::Array{<:Number,2})
        x₂ₜ = (x₂)'
        k = 47
        f = ones(size(x₁,1),size(x₂ₜ,2))
        for i in 1:size(x₁,1)
            for j in 1:size(x₂ₜ,2)
                f[i,j] = -(x₂ₜ[1,j]+k)*(sin(sqrt(abs((x₁[i,1]/2)+(x₂ₜ[1,j]+k)))))-x₁[i,1]*(sin(sqrt(abs(x₁[i,1]-(x₂ₜ[1,j]+k)))))
            end
        end
        surf(x₁,x₂ₜ,f,label="Eggholder Function")
        return f
    end

end
