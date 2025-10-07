#test_ co_kriging

#mle c
hyperparc = [0.1, 2]

hyperparc = convert(Array{Float64}, hyperparc)
pc = hyperparc[size(hyperparc, 1), 1]./5 .+1.4
θc = 10 .^hyperparc[1:(size(hyperparc, 1)-1)] #Searching in a log scale
θc = reshape(θc, size(θc, 1), 1)
Nₛₙc = size(Xₜᵍc, 1)
One = ones(Nₛₙc, 1)
ψₘₐₜc = cormatrix(Xₜᵍc,Xₜᵍc, θc, pc)


hyperparc[1:(size(hyperparc, 1)-1)]


hyperpard = [0.1, 2, 0.1]

#mle d
hyperpard = convert(Array{Float64}, hyperpard) #hyperpard= [θd, pd, ρ]
θd = 10 .^hyperpard[1:(size(hyperpard, 1)-2)] #Searching in a log scale
θd = reshape(θd, size(θd, 1), 1)
pd = hyperpard[size(hyperpard, 1)-1, 1]./5 .+1.4
ρ =  hyperpard[size(hyperpard, 1), 1].*2 .+1 #Rho of the Co-Kriging , searching between [-5 5]
Xₜe = Xₜᵍe
Yₜe = Yₜᵍe
Yₜc = Yₜᵍc # the last Nₛₙe must correspond to the evaluation of the points Xₜe!
Nₛₙe = size(Xₜe, 1) #Number of scenarios
One = ones(Nₛₙe, 1)
ψₘₐₜdXeX = cormatrix(Xₜe,Xₜe, θd, pd)
Ud = cholesky(ψₘₐₜdXeX).U
LnDetψdXe = 2*sum(log.(abs.(diag(Ud))))
d = Yₜe-ρ.*Yₜc[end-(Nₛₙe-1):end]
μₘₗₑd = ((One)'*(Ud\(Ud'\d)))./((One)'*(Ud\(Ud'\One)))
σ²d = ((d-One*μₘₗₑd)'*(Ud\(Ud'\(d-One*μₘₗₑd))))/Nₛₙe
σ²d = diag(σ²d)
LnLd = sum(-1 .*(-(Nₛₙe/2).*log.(σ²d).-LnDetψdXe/2))/Nₛₙe


function Cmatrix(Xₜᵣc::Array{<:Number,2},Xₜᵣe::Array{<:Number,2},θc::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pc::Number,θd::Union{Union{Array{<:Number,2},Array{<:Number,1}},Number},pd::Number,ρ::Number)
    #ψₘₐₜc and σ²c
    Nₛₙc = size(Xₜᵣc, 1)
    Onec = ones(Nₛₙc, 1)
    ψₘₐₜcXcXc = cormatrix(Xₜᵣc, Xₜᵣc, θc, pc)
    Uc = cholesky(ψₘₐₜc).U
    LnDetψc = 2*sum(log.(abs.(diag(Uc))))
    μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜc)))./((Onec)'*(Uc\(Uc'\Onec)))
    σ²c = ((Yₜc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜc-Onec*μₘₗₑc))))/Nₛₙc
    σ²c = diag(σ²c)

    #ψₘₐₜdXeXe and σ²d
    Nₛₙe = size(Xₜᵣe, 1)
    Onee = ones(Nₛₙe, 1)
    ψₘₐₜdXeXe = cormatrix(Xₜᵣe, Xₜᵣe, θd, pd)
    Ud = cholesky(ψₘₐₜdXeXe).U
    LnDetψdXe = 2*sum(log.(abs.(diag(Ud))))
    d = Yₜe-ρ.*Yₜc[end-(Nₛₙe-1):end]
    μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
    σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
    σ²d = diag(σ²d)

    #ψₘₐₜcXeXc and ψₘₐₜcXcXe
    ψₘₐₜcXeXc = cormatrix(Xₜᵣe, Xₜᵣc, θc, pc)
    ψₘₐₜcXcXe = ψₘₐₜcXeXc'
    ψₘₐₜcXeXe = cormatrix(Xₜᵣe, Xₜᵣe, θc, pc)

    #Assembling C
    C = [σ²c*ψₘₐₜcXcXc ρ*σ²c*ψₘₐₜcXcXe; ρ*σ²c*ψₘₐₜcXeXc (ρ^2)*σ²c*ψₘₐₜcXeXe+σ²d*ψₘₐₜdXeXe]
    return C
end
ψₘₐₜcXcXc1 = cormatrix(Xₜᵣc, Xₜᵣc, θc, pc)

#Complete function test with alleatory data
Xₜᵣc = rand(10,1)
Xₜᵣe = rand(4,1)
Xₜᵣc = vcat(Xₜᵣc,Xₜᵣe)
Yₜᵣc = rand(14,1)
Yₜᵣe = rand(4,1)
xₑ=rand(1)[:,:]

θc, pc = hyperpargeneticC(Xₜᵣc,Yₜᵣc)
θc1, pc1 = hyperparNelderMeadC(Xₜᵣc,Yₜᵣc)

θd, pd, ρ = hyperpargeneticD(Xₜᵣe,Yₜᵣe,Yₜᵣc)
θd1, pd1, ρ1 = hyperparNelderMeadD(Xₜᵣe,Yₜᵣe,Yₜᵣc)

θc, pc = 0.801393, 1.5
θd, pd, ρ = 0.1, 1.67, 3
#Cokriging
Nₛₙc = size(Xₜᵣc, 1)
Onec = ones(Nₛₙc, 1)
ψₘₐₜcXcXc = cormatrix(Xₜᵣc, Xₜᵣc, θc, pc)
Uc = cholesky(ψₘₐₜcXcXc).U
μₘₗₑc = ((Onec)'*(Uc\(Uc'\Yₜᵣc)))./((Onec)'*(Uc\(Uc'\Onec)))
σ²c = ((Yₜᵣc-Onec*μₘₗₑc)'*(Uc\(Uc'\(Yₜᵣc-Onec*μₘₗₑc))))/Nₛₙc
σ²c = diag(σ²c)

Nₛₙe = size(Xₜᵣe, 1)
Onee = ones(Nₛₙe, 1)
ψₘₐₜdXeXe = cormatrix(Xₜᵣe, Xₜᵣe, θd, pd)
Ud = cholesky(ψₘₐₜdXeXe).U
d = Yₜᵣe-ρ.*Yₜᵣc[end-(Nₛₙe-1):end]
μₘₗₑd = ((Onee)'*(Ud\(Ud'\d)))./((Onee)'*(Ud\(Ud'\Onee)))
σ²d = ((d-Onee*μₘₗₑd)'*(Ud\(Ud'\(d-Onee*μₘₗₑd))))/Nₛₙe
σ²d = diag(σ²d)

ψₘₐₜcXeXc = cormatrix(Xₜᵣe, Xₜᵣc, θc, pc)
ψₘₐₜcXcXe = ψₘₐₜcXeXc'
ψₘₐₜcXeXe = cormatrix(Xₜᵣe, Xₜᵣe, θc, pc)

C = [σ²c.*ψₘₐₜcXcXc ρ.*σ²c.*ψₘₐₜcXcXe; ρ.*σ²c.*ψₘₐₜcXeXc (ρ^2).*σ²c.*ψₘₐₜcXeXe+σ²d.*ψₘₐₜdXeXe]

ψᵥₑcXcXeval = vectcor(Xₜᵣc, xₑ, θc, pc)
ψᵥₑcXeXeval = vectcor(Xₜᵣe, xₑ, θc, pc)
ψᵥₑdXeXeval = vectcor(Xₜᵣe, xₑ, θd, pd)
