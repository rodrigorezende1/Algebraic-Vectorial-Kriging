"""
This Module contains the evaluation of the function (|x*sin(5*x)|)^x so that a
surrogate model can be evaluated.

This code was developed by: Rodrigo Silva Rezende during his master thesis at
the TU Berlin in the Theoretische Elektrotechnik Department - 25/11/2019
"""

"""
    sinustest(x₁::Array{<:Number,2}, x₂::Array{<:Number,2}, x₃::Number)

Calculates the function (|x*sin(5*x)|)^x for the interval defined by x₁ and x₂.

### Input

    - `x₁` -- downer limit of the interval
    - `x₂` -- upper limit interval of the second variable in the Branin function
    - `x₂` -- number of points inside the limit

### Output

A vector of size = [n₃,1] containing the evaluation of (|x*sin(5*x)|)^x

### Algorithm

The function simply calculates the value of (|x*sin(5*x)|)^x in the range
[x₁,x₂] for n₃ times.

"""
module Sinustest

    using PyPlot
    export sinustest1D

    function sinustest1D(x₁::Number,x₂::Number, x₃::Number)
        if x₂ > x₁ && x₃ >= 3
            x = collect(range(x₁, stop=x₂, length=x₃))
            x = reshape(x, length(x), 1)
            fₒᵤₜ = zeros(size(x,1),1)
            for i in 1:size(x,1)
                fₒᵤₜ[i,1] = (abs(x[i,1]*sin(5*x[i,1])))^x[i,1]
            end
        else
             error("Error: x₂ must be bigger than x₁ and x₃ >= 3")
        end
        return fₒᵤₜ, x
    end
    (fₒᵤₜ,x) = sinustest1D(0,1,20)
    plot(x,fₒᵤₜ)

end
