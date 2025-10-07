"""
This Module contains the evaluation of the Branin function so that surrogate
models can be evaluated.

This code was developed by: Rodrigo Silva Rezende during his master thesis at
the TU Berlin in the Theoretische Elektrotechnik Department - 25/11/2019
"""

module Branin

    using PyPlot
    export Branin_func

    """
        Branin(x₁::Array{<:Number,2},x₂::Array{<:Number,2})

    Calculates the Branin function for the interval defined by x₁ and x₂.

    ### Input

        - `x₁` -- interval of the first variable in the Branin function
        - `x₂` -- interval of the second variable in the Branin function

    ### Output

    A Matrix with the values of the Branin function in each point

    ### Algorithm

    The output Matrix is a i X j matrix containing the evaluations of the Braning
    function on the points [x₁(i);x₂(j)]
    """
    function Branin_func(x₁::Array{<:Number,2},x₂::Array{<:Number,2})
        x₂ₜ = (x₂)'
        a = 1
        b = 1/(4*(π)^2)
        c = 5/(π)
        r = 6
        s = 10
        t = 1/(8*π)
        f = ones(size(x₁,1),size(x₂ₜ,2))
        for i in 1:size(x₁,1)
            for j in 1:size(x₂ₜ,2)
                f[i,j] = a*(x₂ₜ[1,j]-b*(x₁[i,1])^2+c*x₁[i,1]-r)^2+s*(1-t)*cos(x₁[i,1])+s
            end
        end
        surf(x₁,x₂ₜ,f,label="Branin Function")
        return f
    end

end
