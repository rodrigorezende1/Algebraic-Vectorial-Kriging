"""
This module offers the necessary functions to evaluate the global
errors of surrogate methods.
The two methods to avaliate this error are:
    -Root mean square error (RMSE)
    - Correlation Coeficient (r²)

This code was developed by: Rodrigo Silva Rezende during his master thesis at
the TU Berlin in the Theoretische Elektrotechnik Department - 31/01/2020

"""

module ErrorCalc


    export NRMSE, Corcoeff

    """
        NRMSE(Yₜₑₛ::Union{Array{<:Number,2},Number}, yₑ::Union{Array{<:Number,2},Number})

    Calculates the normalized root mean square error of a test data Yₜₑₛ and an
    evaluated data yₑ

    ### Input

    - `Yₜₑₛ` -- Number or array of the testing data
    - `yₑ` -- Number or array of the evaluated data

    ### Output

    A scalar being equal to the NRMSE of both data sets

    ### Notes

    The normalization of the RMSE is done as follows:
    NRMSE = RMSE / (yₘₐₓ-yₘᵢₙ)
     with yₘₐₓ = maximum value of Yₜₑₛ
     and yₘᵢₙ = minimum value of Yₜₑₛ

    ### Algorithm

    The function just performs the following calculation:

        NRMSE =  (ΣΣ((Yₜₑₛ-yₑ)^2)/(Nₛ*Nₒᵤₜ))^(1/2)/(yₘₐₓ-yₘᵢₙ)

        with Nₛ = size(Yₜₑₛ,1)
        and Nₒᵤₜ = size(Yₜₑₛ,2)

    """
    function NRMSE(Yₜₑₛ::Union{Array{<:Number,2},Number},yₑ::Union{Array{<:Number,2},Number})
        Nₜₛ = size(Yₜₑₛ,1) # Nₜₛ - Number of test scenarios
        Nₒᵤₜ = size(Yₜₑₛ,2) #Nₒᵤₜ - Number of outputs pro scenario
        Mat² = (Yₜₑₛ-yₑ).^2
        Vect² = sum(Mat², dims=1)./Nₜₛ
        D = sum(Vect²)/Nₒᵤₜ
        rmse = (D)^(1/2)
        range_Yₜₑₛ = maximum(Yₜₑₛ)-minimum(Yₜₑₛ)
        if size(Yₜₑₛ,1) > 2 || size(Yₜₑₛ,2) > 2
        nmrse = rmse/range_Yₜₑₛ
        else
        nmrse = rmse
        end
        return nmrse

    end

    function Corcoeff(Yₜₑₛ::Union{Array{<:Number,2},Number},yₑ::Union{Array{<:Number,2},Number})
        Nₜₛ = size(Yₜₑₛ,1) # Nₜₛ - Number of test scenarios
        Nₒᵤₜ = size(Yₜₑₛ,2) #Nₒᵤₜ - Number of outputs pro scenario
        Uppart = ((Nₒᵤₜ*Nₜₛ*sum(Yₜₑₛ.*yₑ))-sum(Yₜₑₛ)*sum(yₑ))
        Downpart = sqrt(((Nₒᵤₜ*Nₜₛ*sum((Yₜₑₛ).^2))-(sum(Yₜₑₛ))^2)*((Nₒᵤₜ*Nₜₛ*sum((yₑ).^2))-(sum(yₑ))^2))
        r = (Uppart/Downpart)
        return r
    end

end
