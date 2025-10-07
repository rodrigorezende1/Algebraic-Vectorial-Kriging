"""
This Module contains the evaluation of the functions fe(x) = -x*sin(sqrt(abs(x))),
being the expensive function to be evaluated and the fc(x) = 0.5*fe(x)+0.3*(x-50)+50,
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
module Sinustest_Multi_Fidelity_II

    export sinustestMF_HFII, sinustestMF_LFII


    function sinustestMF_HFII(xᵢₙₜ::Array{<:Number,2})
            xᵢₙₜ = reshape(xᵢₙₜ, length(xᵢₙₜ), 1)
            fₒᵤₜe = zeros(size(xᵢₙₜ,1),1)
            for i in 1:size(xᵢₙₜ,1)
                fₒᵤₜe[i,1] = -xᵢₙₜ[i,1]*sin(sqrt(abs(xᵢₙₜ[i,1])))
            end
        return fₒᵤₜe
    end

    function sinustestMF_LFII(xᵢₙₜ::Array{<:Number,2})
            xᵢₙₜ = reshape(xᵢₙₜ, length(xᵢₙₜ), 1)
            fₒᵤₜc = zeros(size(xᵢₙₜ,1),1)
            for i in 1:size(xᵢₙₜ,1)
                fₒᵤₜc[i,1] = 0.5*(-xᵢₙₜ[i,1]*sin(sqrt(abs(xᵢₙₜ[i,1]))))+0.3*(xᵢₙₜ[i,1]-50)+50
            end
        return fₒᵤₜc
    end

end
