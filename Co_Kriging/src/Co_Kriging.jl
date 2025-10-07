module Co_Kriging

    using LinearAlgebra #Dealing with matricies
    using Optim #MLE Nelder Mead
    using BlackBoxOptim #MLE Genetic
    using SpecialFunctions #Infill Crieria

    export cokriging, hyperpargeneticC, hyperparNelderMeadC, hyperpargeneticD, hyperparNelderMeadD, PredMSEgeneticCOK, PredMSEGNelderMead, ExImpgeneticCOK, ExImpNelderMeadCOK

    #Range MLE = [-2,2]

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
        pc = hyperparc[size(hyperparc, 1), 1]./5 .+1.599 #Searching [0.3999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
        #pc = hyperparc[size(hyperparc, 1), 1]./3 .+1.666666
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
                LnLc = 100.0
        end
        return LnLc
    end

    function hyperpargeneticC(Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2})
            global Xₜᵍc, Yₜᵍc
            Xₜᵍc = Xₜᵣc
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣc, 2)
            res = bboptimize(mleC; SearchRange = (-3.0,2.0), NumDimensions = (Nᵥ+1), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 100.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange its really important! #Searching [0.5999,2.0]
            hyperparc = best_candidate(res)
            hyperparc = reshape(hyperparc, size(hyperparc,1),1)
            θc = 10 .^hyperparc[1:(size(hyperparc,1)-1)]
            θc = reshape(θc, size(θc,1),1)
            pc = hyperparc[size(hyperparc, 1), 1]./5 .+1.599
            #pc = hyperparc[size(hyperparc, 1), 1]./3 .+1.666666
            return θc, pc
    end

    function hyperparNelderMeadC(Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2})
            global Xₜᵍc, Yₜᵍc
            Xₜᵍc = Xₜᵣc
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣc, 2)
            lower = ones((Nᵥ+1),1)*-3
            upper = ones((Nᵥ+1),1)*2.0
            initial_x =ones((Nᵥ+1),1)*-1
            start_time = time()
            result = optimize(mleC, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 100.0)) #time_limit = 100 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            hyperparc = Optim.minimizer(result)
            hyperparc = reshape(hyperparc, size(hyperparc,1),1)
            θc = 10 .^hyperparc[1:(size(hyperparc,1)-1)]
            θc = reshape(θc, size(θc,1),1)
            pc = hyperparc[size(hyperparc, 1), 1]./5 .+1.599
            #pc = hyperparc[size(hyperparc, 1), 1]./3 .+1.666666
            return θc, pc
    end


    function mleD(hyperpard::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
        hyperpard = convert(Array{Float64}, hyperpard) #hyperpard= [θd, pd, ρ]
        θd = 10 .^hyperpard[1:(size(hyperpard, 1)-2)] #Searching in a log scale
        θd = reshape(θd, size(θd, 1), 1)
        pd = hyperpard[size(hyperpard, 1)-1, 1]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
        #pd = hyperpard[size(hyperpard, 1)-1, 1]./3 .+1.666666 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
        #ρ =  hyperpard[size(hyperpard, 1), 1].+7 #Rho of the Co-Kriging , searching between [1 9] is working for all testes until now (29.02.2020)!
        ρ =  hyperpard[size(hyperpard, 1), 1].*2 .+1 #Rho of the Co-Kriging , searching between [-5 5] if the limits are -3,2
        #ρ =  hyperpard[size(hyperpard, 1), 1].*5/4 .+5/2 #searching between [-5 5] if the limits are -2,2
        Xₜᵣe = Xₜᵍe
        Yₜᵣe = Yₜᵍe
        Yₜᵣc = Yₜᵍc # the last Nₛₙe points must correspond to the evaluation of the points Xₜᵣe!
        Nₛₙe = size(Xₜᵣe, 1) #Number of scenarios
        Oned = ones(Nₛₙe, 1)
        ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
        if isposdef(ψₘₐₜdXeXe)
            Ud = cholesky(ψₘₐₜdXeXe).U
            LnDetψdXe = 2*sum(log.(abs.(diag(Ud))))
            d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end,:] #The Nₛₙe - last elements of Yₜᵣc should be the response of the course function to the Xₜᵣe
            μₘₗₑd = ((Oned)'*(Ud\(Ud'\d)))./((Oned)'*(Ud\(Ud'\Oned)))
            σ²d = ((d-Oned*μₘₗₑd)'*(Ud\(Ud'\(d-Oned*μₘₗₑd))))/Nₛₙe
            σ²d = diag(σ²d)
            LnLd = sum(-1 .*(-(Nₛₙe/2).*log.(σ²d).-LnDetψdXe/2))/Nₛₙe #My contribution
        else #Penalty for ill-conditioned
                LnLd = 100.0
            end
        return LnLd
    end


    function hyperpargeneticD(Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},Yₜᵣc::Array{<:Number,2})
            global Xₜᵍe, Yₜᵍe, Yₜᵍc
            Xₜᵍe = Xₜᵣe
            Yₜᵍe = Yₜᵣe
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣe, 2)
            res = bboptimize(mleD; SearchRange = (-3.0,2.0), NumDimensions = (Nᵥ+2),  Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 100.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange its really important! #Searching [0.5999,2.0]
            hyperpard = best_candidate(res)
            hyperpard = reshape(hyperpard, size(hyperpard,1),1)
            θd = 10 .^hyperpard[1:(size(hyperpard, 1)-2)] #Searching in a log scale
            θd = reshape(θd, size(θd, 1), 1)
            pd = hyperpard[size(hyperpard, 1)-1, 1]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
            #pd = hyperpard[size(hyperpard, 1)-1, 1]./3 .+1.666666
            #ρ =  hyperpard[size(hyperpard, 1), 1].+7
            ρ =  hyperpard[size(hyperpard, 1), 1].*2 .+1
            #ρ =  hyperpard[size(hyperpard, 1), 1].*5/4 .+5/2
            return θd, pd, ρ
    end

    function hyperparNelderMeadD(Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},Yₜᵣc::Array{<:Number,2})
            global Xₜᵍe, Yₜᵍe, Yₜᵍc
            Xₜᵍe = Xₜᵣe
            Yₜᵍe = Yₜᵣe
            Yₜᵍc = Yₜᵣc
            Nᵥ = size(Xₜᵣe, 2)
            lower = ones((Nᵥ+2),1)*-3
            upper = ones((Nᵥ+2),1)*2.0
            initial_x =ones((Nᵥ+2),1)*-1
            #initial_x[(Nᵥ+2),1]=1.5 #Special for the Sinus Test Case!!!
            start_time = time()
            result = optimize(mleD, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 100.0)) #time_limit = 100 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            hyperpard = Optim.minimizer(result)
            hyperpard = reshape(hyperpard, size(hyperpard,1),1)
            θd = 10 .^hyperpard[1:(size(hyperpard, 1)-2)] #Searching in a log scale
            θd = reshape(θd, size(θd, 1), 1)
            pd = hyperpard[size(hyperpard, 1)-1, 1]./5 .+1.599 #Searching [0.5999,2.0] the range must have an uppper limit of 2.0 otherwise the result can be completely wrong depending on the number of the sampling data
            #pd = hyperpard[size(hyperpard, 1)-1, 1]./3 .+1.666666
            #ρ =  hyperpard[size(hyperpard, 1), 1].+7
            ρ =  hyperpard[size(hyperpard, 1), 1].*2 .+1
            #ρ =  hyperpard[size(hyperpard, 1), 1].*5/4 .+5/2
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
        σ²c = reshape(σ²c, 1, size(σ²c,1))

        #ψₘₐₜdXeXe and σ²d
        Nₛₙe = size(Xₜᵣe, 1)
        Onee = ones(Nₛₙe, 1)
        ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
        Ud = cholesky(ψₘₐₜdXeXe).U
        d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end,:]
        μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
        σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
        σ²d = diag(σ²d)
        σ²d = reshape(σ²d, 1, size(σ²d,1))

        #ψₘₐₜcXeXc, ψₘₐₜcXcXe and ψₘₐₜcXeXe
        ψₘₐₜcXeXc = cormatrixCoK(Xₜᵣe, Xₜᵣc, θc, pc)
        ψₘₐₜcXcXe = ψₘₐₜcXeXc'
        ψₘₐₜcXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θc, pc)

        #Assembling C
        Nₛₙc = size(Yₜᵣc,1)
        Nₛₙe = size(Yₜᵣe,1)
        Nₒᵤₜ = size(Yₜᵣc,2)
        C = zeros(Nₛₙc+Nₛₙe, Nₛₙc+Nₛₙe, Nₒᵤₜ)
        for i in 1:Nₒᵤₜ
        C[:,:,i] = [σ²c[1,i].*ψₘₐₜcXcXc ρ.*σ²c[1,i].*ψₘₐₜcXcXe; ρ.*σ²c[1,i].*ψₘₐₜcXeXc (ρ^2).*σ²c[1,i].*ψₘₐₜcXeXe+σ²d[1,i].*ψₘₐₜdXeXe]
        end
        #ψᵥₑcXcXeval and ψᵥₑcXeXeval and ψᵥₑdXeXeval
        ψᵥₑcXcXeval = vectcorCoK(Xₜᵣc, xₑ, θc, pc)
        ψᵥₑcXeXeval = vectcorCoK(Xₜᵣe, xₑ, θc, pc)
        ψᵥₑdXeXeval = vectcorCoK(Xₜᵣe, xₑ, θd, pd)

        #Assembling c
        c = zeros(Nₛₙc+Nₛₙe, 1, Nₒᵤₜ)
        for i in 1:Nₒᵤₜ
        c[:,:,i] = [ρ.*σ²c[1,i].*ψᵥₑcXcXeval; (ρ^2).*σ²c[1,i].*ψᵥₑcXeXeval+σ²d[1,i].*ψᵥₑdXeXeval]
        end

        #Assembling μₘₗₑ
        YₜᵣTot = vcat(Yₜᵣc,Yₜᵣe) # YₜᵣTot = [Yₜᵣc; Yₜᵣe]
        NₛₙTot = size(YₜᵣTot, 1)
        OneTot = ones(NₛₙTot, 1)
        μₘₗₑTot = zeros(1,Nₒᵤₜ)
        for i in 1:Nₒᵤₜ
        μₘₗₑTot[1,i] = (OneTot'*inv(C[:,:,i])*YₜᵣTot[:,i]/(OneTot'*inv(C[:,:,i])*OneTot))[1,1]
        end
        #Prediction
        Yₑᵥₐₗ = zeros(1,Nₒᵤₜ)
        for i in 1:Nₒᵤₜ
        Yₑᵥₐₗ[1,i] = (μₘₗₑTot[1,i] + (c[:,:,i]'*inv(C[:,:,i])*(YₜᵣTot[:,i]-OneTot*μₘₗₑTot[1,i]))[1,1])[1,1]
        end

        return Yₑᵥₐₗ
    end

##############Infill Criteria###############################
###################MSE######################################
        function PredMSECOK(xₑ::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
                Nₛₙc = size(Xₜᵣcᵍ,1)
                Nₛₙe = size(Xₜᵣeᵍ,1)
                xₑ =  reshape(xₑ, 1,length(xₑ))
                xₑ = round.(xₑ, digits=4)
                Xₜ_comp = round.(Xₜᵣeᵍ, digits=4)
                if issubset(xₑ,Xₜ_comp) #Points nearby up to 4 digits will be penalized
                    s² = 100.0
                else
                    #ψᵥₑcXcXeval and ψᵥₑcXeXeval and ψᵥₑdXeXeval
                    ψᵥₑcXcXeval = vectcorCoK(Xₜᵣcᵍ, xₑ, θcᵍ, pcᵍ)
                    ψᵥₑcXeXeval = vectcorCoK(Xₜᵣeᵍ, xₑ, θcᵍ, pcᵍ)
                    ψᵥₑdXeXeval = vectcorCoK(Xₜᵣeᵍ, xₑ, θdᵍ, pdᵍ)

                    #Assembling c
                    c = zeros(Nₛₙc+Nₛₙe, 1, Nₒᵤₜᵍ)
                    for i in 1:Nₒᵤₜᵍ
                    c[:,:,i] = [ρᵍ.*σ²cᵍ[1,i].*ψᵥₑcXcXeval; (ρᵍ^2).*σ²cᵍ[1,i].*ψᵥₑcXeXeval+σ²dᵍ[1,i].*ψᵥₑdXeXeval]
                    end

                    #Calculations of the Predicted MSE
                    s² = zeros(1, Nₒᵤₜᵍ)
                    for i in 1:Nₒᵤₜᵍ
                        s²[1,i] = ((ρᵍ^2*σ²cᵍ[:,:,i])+σ²dᵍ[:,:,i].-(c[:,:,i]'*inv(Cᵍ[:,:,i])*c[:,:,i]))[1,1]
                    end
                    s² = sum(s²)
                    s² = -s²
                    return s²
                end
        end


        function PredMSEgeneticCOK(Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2}, Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},θc::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pc::Number,θd::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pd::Number,ρ::Number)
            global Xₜᵣcᵍ,Xₜᵣeᵍ,θcᵍ, pcᵍ,θdᵍ, pdᵍ, ρᵍ, σ²cᵍ, σ²dᵍ, Cᵍ, Nₒᵤₜᵍ
            Xₜᵣcᵍ = Xₜᵣc
            Xₜᵣeᵍ = Xₜᵣe
            θcᵍ = θc
            pcᵍ = pc
            θdᵍ = θd
            pdᵍ = pd
            ρᵍ = ρ

            #ψₘₐₜcXcXc and σ²c
            Nᵥ = size(Xₜᵣc, 2)
            Nₛₙc = size(Xₜᵣc, 1)
            Onec = ones(Nₛₙc, 1)
            ψₘₐₜcXcXc = cormatrixCoK(Xₜᵣc, Xₜᵣc, θc, pc)
            Uc = cholesky(ψₘₐₜcXcXc).U
            μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜᵣc)))./((Onec)'*(Uc\(Uc'\Onec)))
            σ²c = ((Yₜᵣc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜᵣc-Onec*μₘₗₑc))))/Nₛₙc
            σ²c = diag(σ²c)
            σ²c = reshape(σ²c, 1, size(σ²c,1))

            #ψₘₐₜdXeXe and σ²d
            Nₛₙe = size(Xₜᵣe, 1)
            Onee = ones(Nₛₙe, 1)
            ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
            Ud = cholesky(ψₘₐₜdXeXe).U
            d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end,:]
            μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
            σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
            σ²d = diag(σ²d)
            σ²d = reshape(σ²d, 1, size(σ²d,1))

            σ²cᵍ = σ²c
            σ²dᵍ = σ²d

            #ψₘₐₜcXeXc, ψₘₐₜcXcXe and ψₘₐₜcXeXe
            ψₘₐₜcXeXc = cormatrixCoK(Xₜᵣe, Xₜᵣc, θc, pc)
            ψₘₐₜcXcXe = ψₘₐₜcXeXc'
            ψₘₐₜcXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θc, pc)

            #Assembling C
            Nₛₙc = size(Yₜᵣc,1)
            Nₛₙe = size(Yₜᵣe,1)
            Nₒᵤₜ = size(Yₜᵣc,2)
            Nₒᵤₜᵍ = Nₒᵤₜ
            C = zeros(Nₛₙc+Nₛₙe, Nₛₙc+Nₛₙe, Nₒᵤₜ)
            for i in 1:Nₒᵤₜ
            C[:,:,i] = [σ²c[1,i].*ψₘₐₜcXcXc ρ.*σ²c[1,i].*ψₘₐₜcXcXe; ρ.*σ²c[1,i].*ψₘₐₜcXeXc (ρ^2).*σ²c[1,i].*ψₘₐₜcXeXe+σ²d[1,i].*ψₘₐₜdXeXe]
            end
            Cᵍ = C

            res = bboptimize(PredMSECOK; SearchRange = (0.0, 1.0), NumDimensions = (Nᵥ),   Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 100.0, PopulationSize = 50)
            xₑ_best = best_candidate(res)
            xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
            return xₑ_best
        end

        function PredMSEGNelderMead(Xₜ::Array{<:Number,2}, Yₜ::Array{<:Number,2},θ::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},p::Number)
            global Xₜᵣcᵍ,Xₜᵣeᵍ,θcᵍ, pcᵍ,θdᵍ, pdᵍ, ρ, σ²c, σ²d
            Xₜᵣcᵍ = Xₜᵣc
            Xₜᵣeᵍ = Xₜᵣe
            θcᵍ = θc
            pcᵍ = pc
            θdᵍ = θd
            pdᵍ = pd
            ρᵍ = ρ
            σ²cᵍ = σ²c
            σ²dᵍ = σ²d
            #ψₘₐₜcXcXc and σ²c
            Nₛₙc = size(Xₜᵣc, 1)
            Onec = ones(Nₛₙc, 1)
            ψₘₐₜcXcXc = cormatrixCoK(Xₜᵣc, Xₜᵣc, θc, pc)
            Uc = cholesky(ψₘₐₜcXcXc).U
            μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜᵣc)))./((Onec)'*(Uc\(Uc'\Onec)))
            σ²c = ((Yₜᵣc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜᵣc-Onec*μₘₗₑc))))/Nₛₙc
            σ²c = diag(σ²c)
            σ²c = reshape(σ²c, 1, size(σ²c,1))

            #ψₘₐₜdXeXe and σ²d
            Nₛₙe = size(Xₜᵣe, 1)
            Onee = ones(Nₛₙe, 1)
            ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
            Ud = cholesky(ψₘₐₜdXeXe).U
            d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end,:]
            μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
            σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
            σ²d = diag(σ²d)
            σ²d = reshape(σ²d, 1, size(σ²d,1))

            #ψₘₐₜcXeXc, ψₘₐₜcXcXe and ψₘₐₜcXeXe
            ψₘₐₜcXeXc = cormatrixCoK(Xₜᵣe, Xₜᵣc, θc, pc)
            ψₘₐₜcXcXe = ψₘₐₜcXeXc'
            ψₘₐₜcXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θc, pc)

            #Assembling C
            Nₛₙc = size(Yₜᵣc,1)
            Nₛₙe = size(Yₜᵣe,1)
            Nₒᵤₜ = size(Yₜᵣc,2)
            C = zeros(Nₛₙc+Nₛₙe, Nₛₙc+Nₛₙe, Nₒᵤₜ)
            for i in 1:Nₒᵤₜ
            C[:,:,i] = [σ²c[1,i].*ψₘₐₜcXcXc ρ.*σ²c[1,i].*ψₘₐₜcXcXe; ρ.*σ²c[1,i].*ψₘₐₜcXeXc (ρ^2).*σ²c[1,i].*ψₘₐₜcXeXe+σ²d[1,i].*ψₘₐₜdXeXe]
            end

            lower = zeros(Nᵥ,1)
            upper = ones(Nᵥ,1)
            initial_x = ones(Nᵥ,1)*0.5
            start_time = time()
            result = optimize(PredMSECOK, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 100.0)) #time_limit = 100 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            xₑ_best = Optim.minimizer(result)
            xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
            return xₑ_best
        end

########################Exp of Improvement############
        function ExImpCOK(xₑ::Union{Union{Array{<:Number, 2}, Array{<:Number, 1}}, Number})
                Nₛₙc = size(Xₜᵣcᵍ,1)
                Nₛₙe = size(Xₜᵣeᵍ,1)
                xₑ =  reshape(xₑ, 1,length(xₑ))
                xₑ = round.(xₑ, digits=4)
                Xₜ_comp = round.(Xₜᵣeᵍ, digits=4)
                if issubset(xₑ,Xₜ_comp) #Points nearby up to 4 digits will be penalized
                    s² = 100.0
                else
                    #ψᵥₑcXcXeval and ψᵥₑcXeXeval and ψᵥₑdXeXeval
                    ψᵥₑcXcXeval = vectcorCoK(Xₜᵣcᵍ, xₑ, θcᵍ, pcᵍ)
                    ψᵥₑcXeXeval = vectcorCoK(Xₜᵣeᵍ, xₑ, θcᵍ, pcᵍ)
                    ψᵥₑdXeXeval = vectcorCoK(Xₜᵣeᵍ, xₑ, θdᵍ, pdᵍ)

                    #Assembling c
                    c = zeros(Nₛₙc+Nₛₙe, 1, Nₒᵤₜᵍ)
                    for i in 1:Nₒᵤₜᵍ
                    c[:,:,i] = [ρᵍ.*σ²cᵍ[1,i].*ψᵥₑcXcXeval; (ρᵍ^2).*σ²cᵍ[1,i].*ψᵥₑcXeXeval+σ²dᵍ[1,i].*ψᵥₑdXeXeval]
                    end

                    #Prediction
                    NₛₙTot = size(YₜᵣTotᵍ, 1)
                    OneTot = ones(NₛₙTot, 1)
                    Yₑᵥₐₗ = zeros(1,Nₒᵤₜᵍ)
                    for i in 1:Nₒᵤₜᵍ
                    Yₑᵥₐₗ[1,i] = (μₘₗₑTotᵍ[1,i] + (c[:,:,i]'*inv(Cᵍ[:,:,i])*(YₜᵣTotᵍ[:,i]-OneTot*μₘₗₑTotᵍ[1,i]))[1,1])[1,1]
                    end

                    #Calculations of the Predicted MSE
                    s² = zeros(1, Nₒᵤₜᵍ)
                    for i in 1:Nₒᵤₜᵍ
                        s²[1,i] = ((ρᵍ^2*σ²cᵍ[:,:,i])+σ²dᵍ[:,:,i].-(c[:,:,i]'*inv(Cᵍ[:,:,i])*c[:,:,i]))[1,1]
                    end
                    s = (sqrt.(abs.(s²)))

                    #Expected Improvement
                    yₘᵢₙ_e = Yₜᵣeᵍ[argmin(sum(Yₜᵣeᵍ, dims=2))[1],:]' #Summing all the elements of each row and deciding which row has the lowest sum of elements
                    if sum(s²) == 0
                        ExImp = 100.0
                    elseif minimum((1/sqrt(2))*(yₘᵢₙ_e-Yₑᵥₐₗ)./s) < -5
                        xterm1 = ((1/sqrt(2))*(yₘᵢₙ_e-Yₑᵥₐₗ)./s)
                        term = zeros(10,Nₒᵤₜᵍ)
                        for i=1:10
                            term[i,:] = ((-1)^(i-1)*(factorial(2*(i-1))/(2^(i-1)*factorial(i-1)))./(2^(i-1)))*xterm1.^(-(2*(i-1)+1))
                        end
                        B = (yₘᵢₙ_e-Yₑᵥₐₗ)*(1/(2*sqrt(pi)))*sum(term, dims=1)+(s/sqrt(2*π))
                        if minimum(B) > 0
                            ExImp = (log10(2)/log(2))*(log.(B)-(1/2)*((yₘᵢₙ_e-Yₑᵥₐₗ).^2 ./s²)) #abs(B)?
                            ExImp = -sum(ExImp)
                        else
                            ExImp = 100.0
                        end
                    else
                        ExImp1 = (yₘᵢₙ_e-Yₑᵥₐₗ).*(0.5 .+0.5*erf.((yₘᵢₙ_e-Yₑᵥₐₗ)./ (sqrt.(abs.(s²))*sqrt(2)))) #Vectorial Kriging?
                        ExImp2 = s*(1/sqrt(2*π)).*exp.(-(yₘᵢₙ_e-Yₑᵥₐₗ).^2 ./(2*s²))
                        if minimum(ExImp1 + ExImp2) > 0
                        ExImp = -sum(log10.(ExImp1 + ExImp2))
                        else #minimizer
                        ExImp = 100.0
                        end
                    end
                    return ExImp
                end
        end


        function ExImpgeneticCOK(Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2}, Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},θc::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pc::Number,θd::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pd::Number,ρ::Number)
            global Xₜᵣcᵍ,Xₜᵣeᵍ, Yₜᵣeᵍ, θcᵍ, pcᵍ,θdᵍ, pdᵍ, ρᵍ, σ²cᵍ, σ²dᵍ, Cᵍ, Nₒᵤₜᵍ, μₘₗₑTotᵍ, YₜᵣTotᵍ
            Xₜᵣcᵍ = Xₜᵣc
            Xₜᵣeᵍ = Xₜᵣe
            Yₜᵣeᵍ = Yₜᵣe
            θcᵍ = θc
            pcᵍ = pc
            θdᵍ = θd
            pdᵍ = pd
            ρᵍ = ρ

            #ψₘₐₜcXcXc and σ²c
            Nᵥ = size(Xₜᵣc, 2)
            Nₛₙc = size(Xₜᵣc, 1)
            Onec = ones(Nₛₙc, 1)
            ψₘₐₜcXcXc = cormatrixCoK(Xₜᵣc, Xₜᵣc, θc, pc)
            Uc = cholesky(ψₘₐₜcXcXc).U
            μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜᵣc)))./((Onec)'*(Uc\(Uc'\Onec)))
            σ²c = ((Yₜᵣc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜᵣc-Onec*μₘₗₑc))))/Nₛₙc
            σ²c = diag(σ²c)
            σ²c = reshape(σ²c, 1, size(σ²c,1))

            #ψₘₐₜdXeXe and σ²d
            Nₛₙe = size(Xₜᵣe, 1)
            Onee = ones(Nₛₙe, 1)
            ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
            Ud = cholesky(ψₘₐₜdXeXe).U
            d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end,:]
            μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
            σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
            σ²d = diag(σ²d)
            σ²d = reshape(σ²d, 1, size(σ²d,1))

            σ²cᵍ = σ²c
            σ²dᵍ = σ²d

            #ψₘₐₜcXeXc, ψₘₐₜcXcXe and ψₘₐₜcXeXe
            ψₘₐₜcXeXc = cormatrixCoK(Xₜᵣe, Xₜᵣc, θc, pc)
            ψₘₐₜcXcXe = ψₘₐₜcXeXc'
            ψₘₐₜcXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θc, pc)

            #Assembling C
            Nₛₙc = size(Yₜᵣc,1)
            Nₛₙe = size(Yₜᵣe,1)
            Nₒᵤₜ = size(Yₜᵣc,2)
            Nₒᵤₜᵍ = Nₒᵤₜ
            C = zeros(Nₛₙc+Nₛₙe, Nₛₙc+Nₛₙe, Nₒᵤₜ)
            for i in 1:Nₒᵤₜ
            C[:,:,i] = [σ²c[1,i].*ψₘₐₜcXcXc ρ.*σ²c[1,i].*ψₘₐₜcXcXe; ρ.*σ²c[1,i].*ψₘₐₜcXeXc (ρ^2).*σ²c[1,i].*ψₘₐₜcXeXe+σ²d[1,i].*ψₘₐₜdXeXe]
            end
            Cᵍ = C

            #Assembling μₘₗₑ
            YₜᵣTot = vcat(Yₜᵣc,Yₜᵣe)
            YₜᵣTotᵍ = YₜᵣTot
            NₛₙTot = size(YₜᵣTot, 1)
            OneTot = ones(NₛₙTot, 1)
            μₘₗₑTot = zeros(1,Nₒᵤₜ)
            for i in 1:Nₒᵤₜ
            μₘₗₑTot[1,i] = (OneTot'*inv(C[:,:,i])*YₜᵣTot[:,i]/(OneTot'*inv(C[:,:,i])*OneTot))[1,1]
            end
            μₘₗₑTotᵍ = μₘₗₑTot

            res = bboptimize(ExImpCOK; SearchRange = (0.0, 1.0), NumDimensions = (Nᵥ),   Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 100.0, PopulationSize = 50)
            xₑ_best = best_candidate(res)
            xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
            return xₑ_best
        end


        function ExImpNelderMeadCOK(Xₜᵣc::Array{<:Number,2}, Yₜᵣc::Array{<:Number,2}, Xₜᵣe::Array{<:Number,2}, Yₜᵣe::Array{<:Number,2},θc::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pc::Number,θd::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pd::Number,ρ::Number)
            global Xₜᵣcᵍ,Xₜᵣeᵍ, Yₜᵣeᵍ, θcᵍ, pcᵍ,θdᵍ, pdᵍ, ρᵍ, σ²cᵍ, σ²dᵍ, Cᵍ, Nₒᵤₜᵍ, μₘₗₑTotᵍ
            Xₜᵣcᵍ = Xₜᵣc
            Xₜᵣeᵍ = Xₜᵣe
            Yₜᵣeᵍ = Yₜᵣe
            θcᵍ = θc
            pcᵍ = pc
            θdᵍ = θd
            pdᵍ = pd
            ρᵍ = ρ

            #ψₘₐₜcXcXc and σ²c
            Nᵥ = size(Xₜᵣc, 2)
            Nₛₙc = size(Xₜᵣc, 1)
            Onec = ones(Nₛₙc, 1)
            ψₘₐₜcXcXc = cormatrixCoK(Xₜᵣc, Xₜᵣc, θc, pc)
            Uc = cholesky(ψₘₐₜcXcXc).U
            μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜᵣc)))./((Onec)'*(Uc\(Uc'\Onec)))
            σ²c = ((Yₜᵣc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜᵣc-Onec*μₘₗₑc))))/Nₛₙc
            σ²c = diag(σ²c)
            σ²c = reshape(σ²c, 1, size(σ²c,1))

            #ψₘₐₜdXeXe and σ²d
            Nₛₙe = size(Xₜᵣe, 1)
            Onee = ones(Nₛₙe, 1)
            ψₘₐₜdXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θd, pd)
            Ud = cholesky(ψₘₐₜdXeXe).U
            d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end,:]
            μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
            σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
            σ²d = diag(σ²d)
            σ²d = reshape(σ²d, 1, size(σ²d,1))

            σ²cᵍ = σ²c
            σ²dᵍ = σ²d

            #ψₘₐₜcXeXc, ψₘₐₜcXcXe and ψₘₐₜcXeXe
            ψₘₐₜcXeXc = cormatrixCoK(Xₜᵣe, Xₜᵣc, θc, pc)
            ψₘₐₜcXcXe = ψₘₐₜcXeXc'
            ψₘₐₜcXeXe = cormatrixCoK(Xₜᵣe, Xₜᵣe, θc, pc)

            #Assembling C
            Nₛₙc = size(Yₜᵣc,1)
            Nₛₙe = size(Yₜᵣe,1)
            Nₒᵤₜ = size(Yₜᵣc,2)
            Nₒᵤₜᵍ = Nₒᵤₜ
            C = zeros(Nₛₙc+Nₛₙe, Nₛₙc+Nₛₙe, Nₒᵤₜ)
            for i in 1:Nₒᵤₜ
            C[:,:,i] = [σ²c[1,i].*ψₘₐₜcXcXc ρ.*σ²c[1,i].*ψₘₐₜcXcXe; ρ.*σ²c[1,i].*ψₘₐₜcXeXc (ρ^2).*σ²c[1,i].*ψₘₐₜcXeXe+σ²d[1,i].*ψₘₐₜdXeXe]
            end
            Cᵍ = C

            #Assembling μₘₗₑ
            YₜᵣTot = vcat(Yₜᵣc,Yₜᵣe)
            NₛₙTot = size(YₜᵣTot, 1)
            OneTot = ones(NₛₙTot, 1)
            μₘₗₑTot = zeros(1,Nₒᵤₜ)
            for i in 1:Nₒᵤₜ
            μₘₗₑTot[1,i] = (OneTot'*inv(C[:,:,i])*YₜᵣTot[:,i]/(OneTot'*inv(C[:,:,i])*OneTot))[1,1]
            end
            μₘₗₑTotᵍ = μₘₗₑTot

            lower = zeros(Nᵥ,1)
            upper = ones(Nᵥ,1)
            initial_x = ones(Nᵥ,1)*0.5
            start_time = time()
            result = optimize(ExImpCOK, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(show_trace = true, time_limit = 100.0)) #time_limit = 100 seconds, Fminbox is necesseray otherwise it does not respect the upper and lower bounds
            xₑ_best = Optim.minimizer(result)
            xₑ_best = reshape(xₑ_best, 1,length(xₑ_best))
            return xₑ_best
        end

end
