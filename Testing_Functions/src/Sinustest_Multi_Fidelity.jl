"""
This Module contains the evaluation of the functions fe(x) = (6x-2)²*sin(12x-4),
being the expensive function to be evaluated and the fc(x) = A*fₑ(x)+B(x-0.5)-C,
this last one being the course function. With these functions a multi-fidelity
surrogate model can be created and tested. The functions can be either evaluated
together with a fixed number of points using sinustestMF() or separately in a
specified interval using sinustestMF_HF() for the fe(x) and sinustestMF_LF() for
the fc(x).


This code was developed by: Rodrigo Silva Rezende during his master thesis at
the TU Berlin in the Theoretische Elektrotechnik Department - 25/02/2020
"""

"""
    sinustestMF(x₁::Array{<:Number,2}, x₂::Array{<:Number,2}, xₙe::Number,xₙc::Number)

Calculates the functions fe(x) = (6x-2)²*sin(12x-4) and the fc(x) = A*fₑ(x)+B(x-0.5)-C
for the interval defined by x₁ and x₂.

### Input

    - `x₁` -- downer limit of the interval
    - `x₂` -- upper limit interval of the second variable in the Branin function
    - `xₙe` -- number of points inside the limit for the function fe
    - `xₙc` -- number of points inside the limit for the function fc

### Output

Two vectors of size [xₙe,1] and [xₙc,1] containing the evaluation of fe and fc
respectively

### Algorithm

The function simply calculates the values of fe and fc in the range
[x₁,x₂] with xₙe and xₙc points respectively.

"""
module Sinustest_Multi_Fidelity

    export sinustestMF, sinustestMF_HF, sinustestMF_LF

    function sinustestMF(x₁::Number, x₂::Number, xₙe::Number,xₙc::Number)
        if x₂ > x₁ && xₙe >= 3 && xₙc >= 3
            xe = collect(range(x₁, stop=x₂, length=xₙe))
            xe = reshape(xe, length(xe), 1)
            xc = collect(range(x₁, stop=x₂, length=xₙc))
            xc = reshape(xc, length(xc), 1)
            fₒᵤₜe = zeros(size(xe,1),1)
            fₒᵤₜc = zeros(size(xc,1),1)
            A = 0.5
            B = 10
            C = -5
            for i in 1:size(xe,1)
                fₒᵤₜe[i,1] = (6*xe[i,1]-2)^2*sin(12*xe[i,1]-4)
            end
            for i in 1:size(xc,1)
                fₒᵤₜc[i,1] = A*((6*xc[i,1]-2)^2*sin(12*xc[i,1]-4))+B*(xc[i,1]-0.5)+C
            end
        else
             error("Error: x₂ must be bigger than x₁ and x₃ >= 3")
        end
        return fₒᵤₜe, xe, fₒᵤₜc, xc
    end

    function sinustestMF_HF(xᵢₙₜ::Array{<:Number,2})
            xᵢₙₜ = reshape(xᵢₙₜ, length(xᵢₙₜ), 1)
            fₒᵤₜe = zeros(size(xᵢₙₜ,1),1)
            for i in 1:size(xᵢₙₜ,1)
                fₒᵤₜe[i,1] = (6*xᵢₙₜ[i,1]-2)^2*sin(12*xᵢₙₜ[i,1]-4)
            end
        return fₒᵤₜe, xᵢₙₜ
    end

    function sinustestMF_LF(xᵢₙₜ::Array{<:Number,2})
            xᵢₙₜ = reshape(xᵢₙₜ, length(xᵢₙₜ), 1)
            fₒᵤₜc = zeros(size(xᵢₙₜ,1),1)
            A = 0.5
            B = 10
            C = -5
            for i in 1:size(xᵢₙₜ,1)
                fₒᵤₜc[i,1] = A*((6*xᵢₙₜ[i,1]-2)^2*sin(12*xᵢₙₜ[i,1]-4))+B*(xᵢₙₜ[i,1]-0.5)+C
            end
        return fₒᵤₜc, xᵢₙₜ
    end

end
