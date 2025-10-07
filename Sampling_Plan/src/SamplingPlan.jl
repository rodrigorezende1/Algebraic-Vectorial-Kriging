"""
This Module contains all the functions necessary to build the a sampling plan
of the type random latin hyper cube with special "space filling" properties.
It also contains the function to create a fullfactorial sampling plan.
This program is an implementation of the Sampling Plan approach presented in the
Engineering Design via Surrogate Modelling - A.I.J. Forrester et. al. with some
small modifications.

This code was developed by: Rodrigo Silva Rezende during his master thesis at
the TU Berlin in the Theoretische Elektrotechnik Department - 30/01/2020

"""

module SamplingPlan

    using Random
    using LinearAlgebra
    export rlh, jd, mm, mmphi, mmsort, mmphi, perturb, mmlhs, bestrlh, fullfactorial, subsetSP

    """
        function rlh(n::Int64,k::Int64, Edges::Int64=0)

    Generates a random latin hypercube

    ### Input

    - `n` -- Number of desired points in each dimension of the random latin
             hypercube (RLH)
    - `k` -- Number of dimensions or variables of the (RLH)
    - `Egdes` -- Receives "1" if the extrem sampled point will have the center
                 on the edges, otherwise the point will be contained within the
                 domain (Edges=0 default)

    ### Output

    Matrix containing a random latin hypercube

    ### Notes

    The random latin hypercube is created by creating a n X k Matrix in which
    the columns represent the dimensions and each line is a point in the latin
    hyper cube.

    ### Algorithm

    The matrix is created as follows: we fill each column with random permuations
    of {1,2,...,n} until all the matrix is filled. The RLH is normalized into the
    [0,1]ᵏ box.
    This algorithm is the same presented at Engineering Design via Surrogate
    Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function rlh(n::Int64,k::Int64, Edges::Int64=0)
        X = zeros(n, k) # Pre-allocating memory for the sampling plan X
        for i=1:k
            X[:,i] = randperm(n)
        end
        if Edges==1
            X = (X.-1)/(n-1)
        else
            X = (X.-0.5)/(n)
        end
    end

    """
        function jd(X::Array{Float64,2}, p::Int64=1)

    Generates a random latin hypercube

    ### Input

    - `X` -- A random latin hypercube (sampling plan)
    - `p` -- Distance norm (p=1 rectangular - default, p=2 Euclidian)

    ### Output

    - `J` - multiplicity array (the number of pairs separeted by each distance)
    - `distinc_d` - list of distinct distance values
    ### Notes

    The jd() calculates the distance between every two points and then counts
    how many times each distance apperead in the vector of
    distances

    ### Algorithm

    This algorithm is the same as presented in the Engineering Design via
    Surrogate Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function jd(X::Array{Float64,2}, p::Int64=1)

        n = size(X, 1) #Number of points in the sampling plan

        d = zeros(1, convert(Int64,n*(n-1)/2)) #Combination of n points 2 by 2

        for i = 1:n-1 #Compute the distances between all pairs of points
            for j = i+1:n
                d[convert(Int64,(i-1)*n-(i-1)*i/2+j-i)] = sum(abs.(X[i, :].-X[j, :]).^p)^(1/p) # p-norm
            end
        end
        d =  round.(d,sigdigits=7) #Rounding to avoid Float Point Operation discrepancies #My contribution
        distinct_d = sort(unique(d)) #Removing multiple occurences

        J = zeros(size(distinct_d)) #Pre-allocating memory

        for i = 1:size(distinct_d, 1)
            J[i] = sum(in.(distinct_d[i], d))
        end
        return J,distinct_d
    end

    """
        function mm(X1::Array{Float64,2},X2::Array{Float64,2},p::Int64=1)

    Calculates which from both input sampling plans are the best in respect with
    the "space filling" property after the Morris-Mitchel Criteria

    ### Input

    - `X1` -- A first random latin hypercube (sampling plan)
    - `X2` -- A second random latin hypercube (sampling plan)
    - `p` -- Distance norm (p=1 rectangular - default, p=2 Euclidian)

    ### Output

    - `Mmplan` - if Mmplan=0, both have the same property, if Mmplan=1, X1 has
    a better "space filling" property and if Mmplan=2 the second sampling plan
    is a better one

    ### Notes

    The mm() judges which of the two input sampling plan is better with respect
    to the "space fillingnes" after the Morris-Mitchel criteria. If Mmplan=0 it
    does not mean the sampling plans are equally necessary. It just means that
    they have the same property after this criteria.

    ### Algorithm

    This algorithm is the same as presented in the Engineering Design via
    Surrogate Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function mm(X1::Array{Float64,2},X2::Array{Float64,2},p::Int64=1)

        if sortslices(X1, dims=1)==sortslices(X2, dims=1) #Checking wether they have the same points
            Mmplan=0
        else
            (J1, d1) = jd(X1, p) #Calculating the distancies and the multiplicity
            m1 = length(d1)
            (J2, d2) = jd(X2, p)
            m2 = length(d2)

            V1 = zeros(2*m1, 1)
            V2 = zeros(2*m2, 1)

            V1[1:2:2*m1-1] = d1 #Blend the distance and multiplicity
            V1[2:2:2*m1] = -J1 #Maximizing d and minimizing J

            V2[1:2:2*m2-1] = d2
            V2[2:2:2*m2] = -J2
            #At this point Forrester makes a mistake because he writes
            #m = min(m1,m2) and with that he cuts both vectors to the half
            m = min(2*m1,2*m2) #Checking which vector is the shortest one
            V1 = V1[1:m] #Trimming down both vectors to the size of the shortest one
            V2 = V2[1:m]

            c = zeros(m,1) # Generate vector c such that c(i)=1 if V1(i)>V2(i), c(i)=2 if
                            #V1(i)<V2(i) and c(i)=0 otherwise
            for i=1:m
                if V1[i]>V2[i]
                    c[i] = 1
                elseif V1[i]<V2[i]
                    c[i] = 2
                end
            end
            #If the plans are not identical but have the same space–filling
            #properties
            if sum(c) == 0
                Mmplan=0
            else
                i=1
                while c[i] == 0
                    i=i+1
                end
                Mmplan = c[i]
            end
        end
    end

    """
        mmphi(X::Array{Float64,2}, q::Int64=2, p::Int64=1)

    Calculates the value of the θq(X) after the modified Morris-Mitchel criteria

    ### Input

    - `X` -- A random latin hypercube (sampling plan)
    - `q` -- Constant to be used in the calculation of θq(X)
    - `p` -- Distance norm (p=1 rectangular - default, p=2 Euclidian)

    ### Output

    - `θq` - The factor of the modified Morris-Mitchel criteria

    ### Notes

    mmphi() calculates the θq with the fomula:

        θq(X) = (Σⱼᵐ jⱼdⱼ^(-q))^(1/q)

    ### Algorithm

    This algorithm is the same as presented in the Engineering Design via
    Surrogate Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function mmphi(X::Array{Float64,2}, q::Int64=2, p::Int64=1)

        (J, d) = jd(X, p) #Calculating J and d for the X

        Φq = sum(J.*(d.^(-q)))^(1/q) #The sampling plan quality criterion

        return Φq

    end

    """
        mmsort(X3D::Array{Float64, 3}, p::Int64=1)

    Ranks sampling plans according to the Morris-Mitchel criteria

    ### Input

    - `X3D` -- A three dimensional array containing more than one sampling plan
    - `p` -- Distance norm (p=1 rectangular - default, p=2 Euclidian)

    ### Output

    - `Index` - index array containing the ranking from the best (first element)
                to the worst

    ### Notes

    mmsort() uses the function mm() to judge which sampling plan among any two
    sampling plans of the input 3D array has a space "space filling" property.


    ### Algorithm

    This algorithm is the same as presented in the Engineering Design via
    Surrogate Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function mmsort(X3D::Array{Float64, 3}, p::Int64=1)

        Index = collect(1:size(X3D,3))#Pre-allocating memory with the # Xs

        swap_flag = 1 #Buble-sort

        while swap_flag == 1
            swap_flag = 0
            i = 1
            while i <= length(Index)-1
                if mm(X3D[:, :, Index[i]], X3D[:, :, Index[i+1]], p) == 2
                    buffer = Index[i]
                    Index[i] = Index[i+1]
                    Index[i+1] = buffer
                    swap_flag = 1
                end
                i=i+1
            end
        end
        return Index
    end

    """
        mmphisort(X3D::Array{Float64,3}, q::Int64=2, p::Int64=1)

    Ranks sampling plans according to the modified Morris-Mitchel criteria

    ### Input

    - `X3D` -- A three dimensional array containing more than one sampling plan
    - `q` -- Constant to be used in the calculation of θq(X)
    - `p` -- Distance norm (p=1 rectangular - default, p=2 Euclidian)

    ### Output

    - `Index` - index array containing the ranking from the best (first element)
                to the worst

    ### Notes

    phisort() uses the function mmphi() to judge which sampling plan among any two
    sampling plans of the input 3D array has a space "space filling" property.


    ### Algorithm

    This algorithm is the same as presented in the Engineering Design via
    Surrogate Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function mmphisort(X3D::Array{Float64,3}, q::Int64=2, p::Int64=1)

        Index = collect(1:size(X3D,3))#Pre-allocating memory with the # Xs

        swap_flag = 1 #Buble-sort

        while swap_flag == 1
            swap_flag = 0
            i = 1
            while i <= length(Index)-1
                if mmphi(X3D[:, :, Index[i]], q, p) > mmphi(X3D[:, :, Index[i+1]], q, p)
                    buffer = Index[i]
                    Index[i] = Index[i+1]
                    Index[i+1] = buffer
                    swap_flag = 1
                end
                i=i+1
            end
        end
        return Index
    end

    """
        perturb(X::Array{Float64,2}, PertNum::Int64=1)

    Interchanges pairs of randomly chosen elements within randomly chosen columns
    of a sampling plan a number of times. If the input is a RLH the output will
    be as well a RLH.

    ### Input

    - `X` -- An array containing a sampling plan
    - `PertNum` -- Number of pertubations to be made to X

    ### Output

    - `X` - An array containing a sampling plan randomly pertubed by PertNum- times

    ### Notes

    The function just take to elements from different columns and interchanges
    them. It repeats this proceadure PertNum-times

    ### Algorithm

    This algorithm is the same as presented in the Engineering Design via
    Surrogate Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function perturb(X::Array{Float64,2}, PertNum::Int64=1)

        (n, k) = size(X)

        for pert_count = 1:PertNum
            col = convert(Int64,floor((rand(1)*k .+1)[1]))

            el1 = 1 #Choosing two distinct random points
            el2 = 1
            while el1==el2
                el1 = convert(Int64,floor((rand(1)*n .+1)[1]))
                el2 = convert(Int64,floor((rand(1)*n .+1)[1]))
            end
            buffer = X[el1, col] #Swapping the two elements
            X[el1, col] = X[el2, col]
            X[el2, col] = buffer
        end
        return X
    end

    """
        mmlhs(X_start::Array{Float64,2}, population::Int64, iterations::Int64, q::Int64=2)

    Finds the best sampling plan ("space filling") after making a  general search
    in the universe of possible sampling plans with a genetic algorithm. The
    property used to make the comparison is the θq factor of the modified
    Morris-Mitchel criteria

    ### Input

    - `X_start` -- An array with the starting sampling plan (population 0)
    - `population` -- size of the population in each generation
    - `iterations` -- number of times a population should be created and compared
                      also know as number of generation
    - `q` -- Constant to be used in the calculation of θq(X)

    ### Output

    - `X_best` - A sampling plan containing the best sampling plan ("space filling")

    ### Notes

    mmlhs() performs a genetic algorithmic search in order to find the best latin
    hypercube with respect to the θq of the modified Marris-Mitchel criteria.


    ### Algorithm

    This algorithm is the same as presented in the Engineering Design via
    Surrogate Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function mmlhs(X_start::Array{Float64,2}, population::Int64, iterations::Int64, q::Int64=2)

        n = size(X_start,1)

        X_best = X_start[:,:]
        Φ_best = mmphi(X_best)

        leveloff = convert(Int64,floor(0.85*iterations))

        for it = 1:iterations
            if it < leveloff
                mutations = convert(Int64, round(1+(0.5*n-1)*(leveloff-it)/(leveloff-1)))
            else
                mutations = 1
            end
            X_improved = X_best[:,:]
            Φ_improved = Φ_best

            for offspring = 1:population
                X_try = perturb(X_best[:,:], mutations)
                Φ_try = mmphi(X_try, q)

                if Φ_try < Φ_improved
                    X_improved = X_try[:,:]
                    Φ_improved = Φ_try
                end
            end

            if Φ_improved < Φ_best
                X_best = X_improved[:,:]
                Φ_best = Φ_improved
            end
        end
        return X_best
    end

    """
        function bestrlh(n::Int64, k::Int64, population::Int64, iterations::Int64)


    Generates an optimized Latin hypercube by optimizing the Morris-Mitchel criteria
    and also its modified variant for a range of exponents "q"

    ### Input

    - `n` -- Number of desired points in each dimension of the random latin
             hypercube (RLH)
    - `k` -- Number of dimensions or variables of the (RLH)
    - `population` -- size of the population in each generation
    - `iterations` -- number of times a population should be created and compared
                      also know as number of generation

    ### Output

    - `X` - an array containing the optimized sampling plan

    ### Notes

    The bestrlh() uses the function mmlhs() to search the best sampling plans
    for the following q's:
        q = [1 2 5 10 20 50 100]
    after the best X's are found for each q, the best sampling plan among the
    final ones is picked with the help of the Morris-Mitchel criteria (original)

    ### Algorithm

    This algorithm is the same presented at Engineering Design via Surrogate
    Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function bestrlh(n::Int64, k::Int64, population::Int64, iterations::Int64)

        if k < 2
            error("The Latin hypercubes are not defined for k<2")
        end

        q = [1 2 5 10 20 50 100] #Values of qs to optimize Phi_q

        p = 1 #Distance norm = rectangular norm

        X_start = rlh(n,k) #Starting with a random Latin hypercube

        X3D = zeros(n,k,length(q))
        for i = 1:length(q) #For each q optimize X_start with respect to Φ_q
            X3D[:,:,i] = mmlhs(X_start[:,:], population, iterations, q[i])
        end

        Index = mmsort(X3D,p) #Sorting according to the Morris-Mitchell criterion

        Xₒₚₜ = X3D[:,:,Index[1]] #Taking the best X with the Morris-Mitchell criterion

    end

    """
        function fullfactorial(q::Array{Int64,2},Edges::Int64=1)

    Generates a full factorial in the uni cube

    ### Input

    - `q` -- Number of desired points in each dimension
    - `Egdes` -- Receives "1" if the extrem sampled point will have the center
                 on the edges, otherwise the point will be contained within the
                 domain (Edges=0 default)

    ### Output

    - `X` -- full factorial sampling plan

    ### Notes

    Simple generates a fullfactorial sampling plan in the uni cube

    ### Algorithm

    This algorithm is the same presented at Engineering Design via Surrogate
    Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function fullfactorial(q::Array{Int64,2},Edges::Int64=1)

        if minimum(q) < 2
            error("The full factorial must have at least 2 points per dimension")
        end

        n = prod(q) # n is the total number of points in the sampling plan
        k = length(q) # k is the total number of dimensions
        X = zeros(n, k) #pre-allocating memory for the sampling plan X

        for j = 1:k
            if Edges==1
                one_d_slice = collect(range(0, stop=1, step=1/(q[j]-1)))
            else
                one_d_slice = collect(range(1/(2*q[j]), stop=1, step=1/q[j]))
            end
            column = []
            while length(column) < n
                for l=1:q[j]
                    column = [column; ones(prod(q[j+1:k]),1)*one_d_slice[l]]
                end
            end
            X[:,j] = column
        end
        return X
    end

    """
        function subset(X::Array{Int64,2},ns::Int64)

    Generates a subset with optimized sapce-filling properties (as per Morris-Mitchel
    criterion) of a given sampling plan

    ### Input

    - `X` -- Originial sampling plan
    - `ns` -- size of the desired subset

    ### Output

    - `Xₛ` -- subsetwith optimized space-filling properties

    ### Notes

    The norm used here were p = 1 and q = 5. Others norms could be used as well.

    ### Algorithm

    This algorithm is the same presented at Engineering Design via Surrogate
    Modelling - A.I.J. Forrester et. al. with some small modifications.

    """
    function subsetSP(X::Array{Float64,2},ns::Int64)
        n = size(X, 1)
        p = 1
        q = 5
        r = randperm(n)
        Xₛ = X[r[1:ns], :]
        Xᵣ = X[r[ns+1:end], :]

        for j=1:ns
            orig_crit = mmphi(Xₛ,q,p)
            orig_point = Xₛ[j,:]

            #Looking for the best point to substitute the current one with
            bestsub = 1
            bestsubcrit = Inf
            for i=1:n-ns
                #Replacing the current, jth point with each of the remaining one by one
                Xₛ[j, :] = Xᵣ[i, :]
                crit = mmphi(Xₛ, q, p)
                if crit < bestsubcrit
                    bestsubcrit = crit
                    bestsub = i
                end
            end
            if bestsubcrit < orig_crit
                Xₛ[j, :] = Xᵣ[bestsub, :]
            else
                Xₛ[j, :] = orig_point
            end
        end
        return Xₛ
    end

end
