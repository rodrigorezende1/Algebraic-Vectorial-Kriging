#Here starts the end - Co-Kriging
module Co_Kriging_1D

    using LinearAlgebra #Dealing with matricies
    using Optim #MLE Nelder Mead
    using BlackBoxOptim #MLE Genetic

    export cokriging, hyperpargeneticC, hyperparNelderMeadC, hyperpargeneticD, hyperparNelderMeadD

    function cormatrixCoK(Xₜ₁::Union{Array{<:Number,2},Number},Xₜ₂::Union{Array{<:Number,2},Number},θ::Union{Array{<:Number,2},Number},p::Number)
        ψₘₐₜ = zeros(size(Xₜ₁, 1), size(Xₜ₂, 1))
        for i in 1:size(Xₜ₁, 1)
            for j in 1:size(Xₜ₂, 1)
                ψₘₐₜ[i, j] = exp(-sum(θ.*((abs.(Xₜ₁[i, :]-Xₜ₂[j, :])).^p))) #Function abs() is really important here since p is not necessarily 2
            end
        end
        #ψₘₐₜ = ψₘₐₜ+I*1e-50 # This command is not possible anymore since we are dealing also with non square matrixes #Constructing ψₘₐₜ and adding a small number to reduce ill conditioning
        return ψₘₐₜ
    end

    function vectcorCoK(Xₜ::Array{<:Number,2},xₑ::Union{Array{<:Number,2},Number},θ::Union{Array{<:Number,2},Number},p::Number) #Here xₑ is every element of Xₑ. So this calculation is done for every evaluation point
        ψₑᵥₐₗ = zeros(size(Xₜ, 1),1)
        for i in 1:size(Xₜ, 1)
            ψₑᵥₐₗ[i,1] = exp(-sum(θ.*((abs.(xₑ'.-Xₜ[i, :])).^p))) #The .- is necessary since Xₜ[i, :]::Array{Float64,1}
        end
        return ψₑᵥₐₗ
    end


    #Starting with the MLE of the course data

    function mleC(hyperparc::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
        hyperparc = convert(Array{Float64}, hyperparc)
        pc = hyperparc[size(hyperparc, 1), 1]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
        θc = 10 .^hyperparc[1:(size(hyperparc, 1)-1)] #Searching in a log scale
        θc = reshape(θc, size(θc, 1), 1)
        Xₜᵣc = Xₜᵍc
        Yₜᵣc = Yₜᵍc
        Nₛₙc = size(Xₜᵣc, 1)
        Onec = ones(Nₛₙc, 1)
        ψₘₐₜcXcXc = cormatrixCoK(Xₜᵣc, Xₜᵣc, θc, pc)
        if isposdef(ψₘₐₜcXcXc)
            Uc = cholesky(ψₘₐₜcXcXc).U
            LnDetψc = 2*sum(log.(abs.(diag(Uc))))
            μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜᵣc)))./((Onec)'*(Uc\(Uc'\Onec)))
            σ²c = ((Yₜᵣc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜᵣc-Onec*μₘₗₑc))))/Nₛₙc
            σ²c = diag(σ²c)
            LnLc = sum(-1 .*(-(Nₛₙc/2).*log.(σ²c).-LnDetψc/2))/Nₛₙc
        else #Penalty for ill-conditioned
                LnLc = 10000.0
        end
        return LnLc
    end

    function hyperpargeneticC(Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2})
            global Xₜᵍc, Yₜᵍc
            Xₜᵍc = Xₜᵣc
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣc, 2)
            res = bboptimize(mleC; SearchRange = (-6.0, 2.0), NumDimensions = (Nᵥ+1), MaxTime = 30.0, PopulationSize = 50000) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange its really important! #Searching [0.5999,2.0]
            hyperparc = best_candidate(res)
            hyperparc = reshape(hyperparc, size(hyperparc,1),1)
            θc = 10 .^hyperparc[1:(size(hyperparc,1)-1)]
            θc = reshape(θc, size(θc,1),1)
            pc = hyperparc[size(hyperparc, 1), 1]./5 .+1.599
            return θc, pc
    end

    function hyperparNelderMeadC(Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2})
            global Xₜᵍc, Yₜᵍc
            Xₜᵍc = Xₜᵣc
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣc, 2)
            lower = ones((Nᵥ+1),1)*-6
            upper = ones((Nᵥ+1),1)*2
            initial_x =ones((Nᵥ+1),1)*-1
            start_time = time()
            result = optimize(mleC, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(time_limit = 30.0)) #time_limit = 30 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            hyperparc = Optim.minimizer(result)
            hyperparc = reshape(hyperparc, size(hyperparc,1),1)
            θc = 10 .^hyperparc[1:(size(hyperparc,1)-1)]
            θc = reshape(θc, size(θc,1),1)
            pc = hyperparc[size(hyperparc, 1), 1]./5 .+1.599
            return θc, pc
    end


    function mleD(hyperpard::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
        hyperpard = convert(Array{Float64}, hyperpard) #hyperpard= [θd, pd, ρ]
        θd = 10 .^hyperpard[1:(size(hyperpard, 1)-2)] #Searching in a log scale
        θd = reshape(θd, size(θd, 1), 1)
        pd = hyperpard[size(hyperpard, 1)-1, 1]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
        ρ =  hyperpard[size(hyperpard, 1), 1].+7 #Rho of the Co-Kriging , searching between [1 9] is working for all testes until now (29.02.2020)!
        #ρ =  hyperpard[size(hyperpard, 1), 1].*2 .+1 #Rho of the Co-Kriging , searching between [-5 5]
        Xₜᵣe = Xₜᵍe
        Yₜᵣe = Yₜᵍe
        Yₜᵣc = Yₜᵍc # the last Nₛₙe points must correspond to the evaluation of the points Xₜᵣe!
        Nₛₙe = size(Xₜᵣe, 1) #Number of scenarios
        Oned = ones(Nₛₙe, 1)
        ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
        if isposdef(ψₘₐₜdXeXe)
            Ud = cholesky(ψₘₐₜdXeXe).U
            LnDetψdXe = 2*sum(log.(abs.(diag(Ud))))
            d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end] #The Nₛₙe - last elements of Yₜᵣc should be the response of the course function to the Xₜᵣe
            μₘₗₑd = ((Oned)'*(Ud\(Ud'\d)))./((Oned)'*(Ud\(Ud'\Oned)))
            σ²d = ((d-Oned*μₘₗₑd)'*(Ud\(Ud'\(d-Oned*μₘₗₑd))))/Nₛₙe
            σ²d = diag(σ²d)
            LnLd = sum(-1 .*(-(Nₛₙe/2).*log.(σ²d).-LnDetψdXe/2))/Nₛₙe #My contribution
        else #Penalty for ill-conditioned
                LnLd = 10000.0
            end
        return LnLd
    end


    function hyperpargeneticD(Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},Yₜᵣc::Array{<:Number,2})
            global Xₜᵍe, Yₜᵍe, Yₜᵍc
            Xₜᵍe = Xₜᵣe
            Yₜᵍe = Yₜᵣe
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣe, 2)
            res = bboptimize(mleD; SearchRange = (-6.0, 2.0), NumDimensions = (Nᵥ+2), MaxTime = 30.0, PopulationSize = 50000) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange its really important! #Searching [0.5999,2.0]
            hyperpard = best_candidate(res)
            hyperpard = reshape(hyperpard, size(hyperpard,1),1)
            θd = 10 .^hyperpard[1:(size(hyperpard, 1)-2)] #Searching in a log scale
            θd = reshape(θd, size(θd, 1), 1)
            pd = hyperpard[size(hyperpard, 1)-1, 1]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
            ρ =  hyperpard[size(hyperpard, 1), 1].+7
            #ρ =  hyperpard[size(hyperpard, 1), 1].*2 .+1
            return θd, pd, ρ
    end

    function hyperparNelderMeadD(Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},Yₜᵣc::Array{<:Number,2})
            global Xₜᵍe, Yₜᵍe, Yₜᵍc
            Xₜᵍe = Xₜᵣe
            Yₜᵍe = Yₜᵣe
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣe, 2)
            lower = ones((Nᵥ+2),1)*-6
            upper = ones((Nᵥ+2),1)*2
            initial_x =ones((Nᵥ+2),1)*-1
            #initial_x[(Nᵥ+1),1]=1.5 #Special for the Sinus Test Case!!!
            start_time = time()
            result = optimize(mleD, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(time_limit = 30.0)) #time_limit = 30 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            hyperpard = Optim.minimizer(result)
            hyperpard = reshape(hyperpard, size(hyperpard,1),1)
            θd = 10 .^hyperpard[1:(size(hyperpard, 1)-2)] #Searching in a log scale
            θd = reshape(θd, size(θd, 1), 1)
            pd = hyperpard[size(hyperpard, 1)-1, 1]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
            ρ =  hyperpard[size(hyperpard, 1), 1].+7
            #ρ =  hyperpard[size(hyperpard, 1), 1].*2 .+1
            return θd, pd, ρ
    end

    # cokriging(xₑ::Union{Array{<:Number,2},Number},Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2}, Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},θc::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pc::Number,θd::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pd::Number,ρ::Number)

    function cokriging(xₑ::Union{Array{<:Number,2},Number},Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2}, Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},θc::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pc::Number,θd::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pd::Number,ρ::Number)
        #ψₘₐₜcXcXc and σ²c
        Nₛₙc = size(Xₜᵣc, 1)
        Onec = ones(Nₛₙc, 1)
        ψₘₐₜcXcXc = cormatrixCoK(Xₜᵣc, Xₜᵣc, θc, pc)
        Uc = cholesky(ψₘₐₜcXcXc).U
        μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜᵣc)))./((Onec)'*(Uc\(Uc'\Onec)))
        σ²c = ((Yₜᵣc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜᵣc-Onec*μₘₗₑc))))/Nₛₙc
        σ²c = diag(σ²c)


        #ψₘₐₜdXeXe and σ²d
        Nₛₙe = size(Xₜᵣe, 1)
        Onee = ones(Nₛₙe, 1)
        ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
        Ud = cholesky(ψₘₐₜdXeXe).U
        d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end]
        μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
        σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
        σ²d = diag(σ²d)

        #ψₘₐₜcXeXc, ψₘₐₜcXcXe and ψₘₐₜcXeXe
        ψₘₐₜcXeXc = cormatrixCoK(Xₜᵣe, Xₜᵣc, θc, pc)
        ψₘₐₜcXcXe = ψₘₐₜcXeXc'
        ψₘₐₜcXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θc, pc)

        #Assembling C
        C = [σ²c.*ψₘₐₜcXcXc ρ.*σ²c.*ψₘₐₜcXcXe; ρ.*σ²c.*ψₘₐₜcXeXc (ρ^2).*σ²c.*ψₘₐₜcXeXe+σ²d.*ψₘₐₜdXeXe]

        #ψᵥₑcXcXeval and ψᵥₑcXeXeval and ψᵥₑdXeXeval
        ψᵥₑcXcXeval = vectcorCoK(Xₜᵣc, xₑ, θc, pc)
        ψᵥₑcXeXeval = vectcorCoK(Xₜᵣe, xₑ, θc, pc)
        ψᵥₑdXeXeval = vectcorCoK(Xₜᵣe, xₑ, θd, pd)

        #Assembling c
        c = [ρ.*σ²c.*ψᵥₑcXcXeval; (ρ^2).*σ²c.*ψᵥₑcXeXeval+σ²d.*ψᵥₑdXeXeval]

        #Assembling μₘₗₑ
        YₜᵣTot = vcat(Yₜᵣc,Yₜᵣe) # YₜᵣTot = [Yₜᵣc; Yₜᵣe]
        NₛₙTot = size(YₜᵣTot, 1)
        OneTot = ones(NₛₙTot, 1)
        μₘₗₑTot = OneTot'*inv(C)*YₜᵣTot/(OneTot'*inv(C)*OneTot)

        #Prediction
        Yₑᵥₐₗ = μₘₗₑTot + c'*inv(C)*(YₜᵣTot-OneTot*μₘₗₑTot)

        return Yₑᵥₐₗ
    end

end
