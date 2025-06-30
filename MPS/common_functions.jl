using ITensors
using ITensorMPS
using Kronecker # has the same convention as Mathematica
using LinearAlgebra
using Plots
# using ITensorMPS.HDF5
using JLD
using DelimitedFiles
using LaTeXStrings
using DelimitedFiles
using HDF5
# Define projectors:

const Id = [1 0; 0 1];
const X = [0 1 ; 1 0];
const Y = [0 -1im ; 1im 0];
const Z = [1 0 ; 0 -1];
const Hadamard = 1/sqrt(2)*[1 1; 1 -1];
const Y_tilde = [0 1; -1 0]; # Y_tilde = -XZ

linspace(start, stop, n) = [start + (i-1)*(stop - start)/(n-1) for i in 1:n]
si(n) = siteinds("S=1/2", n)

p00 = (Id + Z)/2; # = |0><0|
p11 = (Id - Z)/2; # = |1><1|
ppp = (Id + X)/2; # = |+><+|, p stands for plus
pmm = (Id - X)/2; # = |-><-|, m stands for minus
p01 = (X + 1im *Y)/2; # = |0><1|
p10 = (X - 1im *Y)/2; # = |1><0|
pmp = [1 1; -1 -1];
ppm = [1 -1; 1 -1];
proj=[p00, p01, p10, p11, ppp, pmm]

proj_d =[proj[i]' for i in 1:length(proj)];
q00, q01, q10, q11, qpp, qmm = proj_d;

Id_tensor(leg1, leg2) = ITensor(Matrix(I, dim(leg1), dim(leg2)), leg1, leg2)

kronr(args...) = kron(reverse(args)...)

function apply_op(state::MPS, op, i)
    ψ = deepcopy(state)
    op_tensor = ITensor(op, siteinds(ψ)[i], siteinds(ψ)[i]')
    newA = op_tensor * ψ[i]
    noprime!(newA)
    ψ[i] = newA
    return ψ
end


function load_state(filename)
    f = h5open(filename,"r")
    read_state = read(f,"psi",MPS)
    close(f)
    return read_state
end

function linear_fit(x, y)
    Lx = length(x)
    Ly = length(y)
    L = Lx
    if Lx != Ly
        println("X and Y have different dimensions")
    end
    x_bar = sum(x)/L
    y_bar = sum(y)/L
    b = sum( (x .- x_bar) .* (y .- y_bar) ) / sum( (x .- x_bar) .^ 2 )
    a = y_bar - b * x_bar
    return [a, b]
end 

function print_matrix(M)
    for i in 1:size(M)[1]
        for j in 1:size(M)[2]
            print(round( M[i,j]; digits = 5))
            print("\t")
            
        end
        println()
    end
end
function directsum(A::Matrix{Float64}, B::Matrix{Float64})
    size1 = size(A)[1]
    size2 = size(B)[1]
    sz = size1+ size2
    M = fill(0.0, (sz, sz))
    for i in 1:size1
        for j in 1:size1
            M[i,j] = A[i,j]
        end
    end
    for i in 1:size2
        for j in 1:size2
            M[i+size1, j+size1] = B[i,j]
        end
    end
    return M
end

function hermiticity(state)
    s = siteinds(state)
    swp = swapMPO(s)
    return norm( conj(apply(swp, state)) - state)
end

unitary_distance(A) = norm(A'*A-I)

function directsum(v1::Vector{Float64}, v2::Vector{Float64})
    return vcat(v1, v2)
end

vv_to_matrix(x) = mapreduce(permutedims, vcat, x)

function get_ψid(s::Vector{Index{Int64}})
    ψ = MPS(s, linkdims=[i%2 == 1 ? 2 : 1 for i in 1:length(s)-1])
    l = linkinds(ψ)

    ψ[1] = ITensor(Id, s[1], l[1])
    for i in 2:length(s)-1
        ψ[i] = ITensor(Id, l[i-1], l[i], s[i])
    end
    ψ[end] = ITensor(Id, l[end], s[end])
    return ψ
end

function traceMPS(state)
    s = siteinds(state)
    ψid = get_ψid(s)
    return inner(ψid, state)
end

function get_ρ_trivial(s)

    ρ_trivial = MPS(s, linkdims=1)
    l = linkinds(ρ_trivial)

    plus = [1., 1.]/√2

    ρ_trivial[1] = ITensor( plus, s[1], l[1])
    ρ_trivial[end] = ITensor(plus , s[end], l[end])
    for i in 2:length(ρ_trivial)-1
        ρ_trivial[i] = ITensor(plus, l[i-1], l[i], s[i])
    end
  return ρ_trivial
end

function multiply(ψ::MPS, operator::Matrix{Int64}, a::Int64)
    state = deepcopy(ψ)
    s = siteinds(state)
    On = ITensor(operator, s[a],s[a]' )
    new_n = state[a] * On
    noprime!(new_n)
    state[a] = new_n
    return state
end

function two_qubit_swap(sites, a, b)
    os = OpSum()
    os += 1/2, Z, a, Z, b
    os += 1/2, Id, a, Id, b
    os += 1/2, X, a, X, b
    os += -1/2, Y_tilde, a, Y_tilde, b
   return MPO(os, sites)
end

function swapMPO(sites::Vector{Index{Int64}})
    N::Int64 = length(sites) ÷ 2
    temp::MPO = two_qubit_swap(sites, 1,2)
    for i in 2:N
        temp = apply(two_qubit_swap(sites, 2i-1, 2i), temp)
    end
    return temp
end

function purity(ψ)
    denominator = (traceMPS(ψ))^2
    return inner(ψ', swapMPO(siteinds(ψ)), ψ)/denominator
end

function get_symmetry_op(sites)
    N = length(sites)
    odd_left = [ i%4 == 1 ? X : Id for i in 1:2*N]
    odd_right = [ i%4 == 2 ? X : Id for i in 1:2*N]
    even_left = [ i%4 == 3 ? X : Id for i in 1:2*N]
    even_right = [ i%4 == 0 ? X : Id for i in 1:2*N]
    Uo_left = MPO(sites, odd_left);
    Uo_right = MPO(sites, odd_right);
    Ue_left = MPO(sites, even_left);
    Ue_right = MPO(sites, even_right);
    return Uo_left, Uo_right, Ue_left, Ue_right
end

function symm_randomMPS(sites, χ = 6)
    Uo_left, Uo_right, Ue_left, Ue_right = get_symmetry_op(sites)
    ψ = randomMPS(sites, χ)
    ψ = 1/2*( ψ + apply(Uo_left, apply(Uo_right,ψ)))
    ψ = 1/2*( ψ + apply(Ue_left,ψ))
    ψ = 1/2*( ψ + apply(Ue_right,ψ))
    return ψ
end

function plusplus_project(ϕ)
    sites = siteinds(ϕ)
    Uo_left, Uo_right, Ue_left, Ue_right = get_symmetry_op(sites)
    ψ = 1/2*( ϕ + apply(Uo_left, apply(Uo_right,ϕ)))
    ψ = 1/2*( ψ + apply(Ue_left,ψ))
    ψ = 1/2*( ψ + apply(Ue_right,ψ))
    return ψ/norm(ψ)
end

function project_strong(ϕ)
    sites = siteinds(ϕ)
    Uo_left, Uo_right, Ue_left, Ue_right = get_symmetry_op(sites)
    ψ = ϕ
    ψ = 1/2*( ψ + apply(Ue_left,ψ))
    ψ = 1/2*( ψ + apply(Ue_right,ψ))
    return ψ/norm(ψ)
end

# -------------------------- get ρ cluster --------------------------

Uo_short = [1.0 0.0; 0.0 0.0;;; 0.0 0.0; 0.0 1.0;;; 0.0 0.0; 1.0 0.0;;; 0.0 1.0; 0.0 0.0]
Vo_short = [1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0;;; 0.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
Ue_short = [0.5 0.5; -0.5 0.5;;; -0.5 0.5; 0.5 0.5;;; -0.5 -0.5; -0.5 0.5;;; -0.5 0.5; -0.5 -0.5];
Ve_short = [0.0 0.0; 0.0 0.0; -1.0 0.0; 0.0 -1.0;;; 1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0];

# Convention t[i,j,k,l] = ∑_a U(k,i,a) *V(a,l,j)
#     k               k     l
#     |               |     | 
# -i--t--j-   ==  -i--U--a--V--j-
#     |
#     l
Uo = fill(0.,2, 4, 4)
Ue = fill(0.,2, 4, 4)
for i in 1:2
    for k in 1:2
        for a in 1:4
            Uo[k,i,a] = Uo_short[k,i,a]
            Ue[k,i,a] = Ue_short[k,i,a]
        end
    end
end

Vo = fill(0.,4, 2, 4)
Ve = fill(0.,4, 2, 4)
for a in 1:4
    for l in 1:2
        for j in 1:2
            Vo[a,l,j] = Vo_short[a,l,j]
            Ve[a,l,j] = Ve_short[a,l,j]
        end
    end
end

Vez = fill(0.,4,2,4)
for a in 1:4
    for l in 1:2    
        for j in 1:2
            for l_prime in 1:2
                Vez[a, l, j] += Ve[a, l_prime, j] * Z[l_prime, l]
            end
        end
    end
end



function get_ρcluster(bc_left, bc_right, last_tensor, sites)  
    ρcluster = MPS(sites;linkdims=4)
    s = siteinds(ρcluster)
    lnks = linkinds(ρcluster)
    N = length(sites) ÷ 2
    ρcluster[1] = ITensor(Uo[:,bc_left,:], s[1], lnks[1])
    for i in 2:2*N-1
        if i%4 == 2
            ρcluster[i] = ITensor(Vo, lnks[i-1], s[i], lnks[i])
        elseif i%4 == 3
            ρcluster[i] = ITensor(Ue, s[i], lnks[i-1], lnks[i])
        elseif i%4 == 0
            ρcluster[i] = ITensor(Ve, lnks[i-1], s[i], lnks[i])
        elseif i%4 == 1
            ρcluster[i] = ITensor(Uo, s[i], lnks[i-1], lnks[i])
        else
            println("Error")
        end
    end
    ρcluster[2*N] = ITensor(last_tensor[:,:,bc_right], lnks[2*N-1], s[2*N])
    return ρcluster
end



function get_eight_states(sites)
    ρplain = [ [get_ρcluster(i,j, Ve, sites) for i in 1:2] for j in 1:2]
    ρplain = vcat(ρplain...)
    ρz = [ [get_ρcluster(i,j, Vez, sites) for i in 1:2] for j in 1:2]
    ρz = vcat(ρz...)
    ρ = vcat(ρplain, ρz)
    # println("Not symmetric ρ")
    ρ = [ρ[1]+ρ[4], ρ[1]-ρ[4], ρ[2]+ρ[3], ρ[2]-ρ[3], ρ[5]+ρ[8], ρ[5]-ρ[8], ρ[6]+ρ[7], ρ[6]-ρ[7]] # These are symmetric states
    ρ = [ρ[i] / norm(ρ[i]) for i in 1:8] # Normalize
    # ρ = vcat([ρ[i] + ρ[i+4] for i in 1:4], [ρ[i] - ρ[i+4] for i in 1:4]) # Making the states traceful 
    # ρ = [i%2 == 1 ? ρ[(i+1)÷ 2] : ρ[i÷2 + 4] for i in 1:8]
    return ρ
end
;

function symmetry_charge(r, ϵ=1e-5)
    sites = siteinds(r)
    Uo_left, Uo_right, Ue_left, Ue_right = get_symmetry_op(sites)
    n1 = norm(apply(Ue_left, r) - r)
    n2 = norm(apply(Ue_left, r) + r)
    n3 = norm(apply(Uo_right, apply(Uo_left, r)) - r)
    n4 = norm(apply(Uo_right, apply(Uo_left, r)) + r)
    n5 = norm(apply(Ue_right, r) - r)
    n6 = norm(apply(Ue_right, r) + r)

    c1, c15, c2 = "0", "0", "0"
    n1 < ϵ ? c1="+" : nothing
    n2 < ϵ ? c1="-" : nothing
    n3 < ϵ ? c2="+" : nothing
    n4 < ϵ ? c2="-" : nothing
    n5 < ϵ ? c15="+" : nothing
    n6 < ϵ ? c15="-" : nothing
    println("Charges are (even left, even right, odd) = ("*c15*", "*c1*", "*c2*")")
    return c1, c15, c2
end

function project(ϕ)
    sites = siteinds(ϕ)
    Uo_left, Uo_right, Ue_left, Ue_right = get_symmetry_op(sites)
    Uo = apply(Uo_right, Uo_left)
    projected = []
    for s1 in [1,-1]
        ψ1 = 1/2*( ϕ +s1* apply(Uo,ϕ))
        for s2 in [1,-1]
            ψ2 = 1/2*( ψ1 + s2*apply(Ue_left,ψ1))
            for s3 in [1,-1]
                ψ3 = 1/2*( ψ2 + s3*apply(Ue_right,ψ2))
                den = norm(ψ3)
                if den < 1e-4
                    println("Small projection in $s1, $s2, $s3 sector = ", den)
                else
                    push!(projected, deepcopy(ψ3)/den)
                end
            end
        end
    end
    return projected
end



# -------------- Are the density matrices legitimate  -------------------------


function check_positivity(psi, χ = 5)
    s2 = siteinds(psi)
    N = length(s2) ÷ 2
    rarray = [i==1 || i==N ? 2 .* rand(2, χ) .- 1 + 2im .* rand(2, χ) .- 1im :  2 .* rand(2, χ,χ) .- 1 + 2im .* rand(2, χ, χ) .- 1im for i in 1:N]

    l_left = [Index(χ, "l_left"*string(i)) for i in 1:N-1] 
    t1_left = ITensor(rarray[1], s2[1], l_left[1])
    t_left = [t1_left]
    for j in 2: N-1
        push!(t_left, ITensor(rarray[j], s2[2j-1], l_left[j-1], l_left[j]) )
    end
    push!(t_left, ITensor(rarray[N], s2[2N-1], l_left[N-1]) )


    l_right = [Index(χ, "l_right"*string(i)) for i in 1:N-1] 
    t1_right = ITensor(conj(rarray[1]), s2[2], l_right[1])
    t_right = [t1_right]
    for j in 2: N-1
        push!(t_right, ITensor(conj(rarray[j]), s2[2j], l_right[j-1], l_right[j]) )
    end
    push!(t_right, ITensor(conj(rarray[N]), s2[2N], l_right[N-1]) )

    # contraction

    

    L = ITensor(1.)
    for i in 1:N
        L *= psi[2i-1]
        L *= t_left[i]
        L *= psi[2i]
        L *= t_right[i]
    end
    return L[1]
end

function check_positivity_multiple(ρ, χ=2, N=100, ϵ=1e-10)
    for i in 1:N
        exp_val = check_positivity(ρ, χ)
        if abs(imag(exp_val)) > ϵ
            println("Error. Imaginary ", exp_val)
            return false
        end
        if real(exp_val) < -ϵ
            println("Error. Negative ", exp_val)
            return false
        end
    end
    return true
end 

function L01( i, proj=[p00, p01, p10, p11, ppp, pmm], coeff=1)
    if coeff != 0.0
        println("L0 old might have a bug. Check odd symmetry")
    end
    os = OpSum()
    p00, p01, p10, p11, ppp, pmm = proj
    # Applies the opreator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    
    # Terms below add L_0^1 ⊗ (L_0^1)^*
    os += coeff, p00, 2*i-3, p00, 2*i-2, ppp, 2*i-1, ppp, 2*i, p01, 2*i+1, p01, 2*i+2
    os += coeff, p00, 2*i-3, p01, 2*i-2, ppp, 2*i-1, ppp, 2*i, p01, 2*i+1, p00, 2*i+2
    os += coeff, p01, 2*i-3, p00, 2*i-2, ppp, 2*i-1, ppp, 2*i, p00, 2*i+1, p01, 2*i+2
    os += coeff, p01, 2*i-3, p01, 2*i-2, ppp, 2*i-1, ppp, 2*i, p00, 2*i+1, p00, 2*i+2
    
    # Terms below add -1/2 (L_0^1^\dagger L_0^1) ⊗ I --> Operates only on L <=> odd
    os += -1/2*coeff, p00, 2*i-3, ppp, 2*i-1, p11, 2*i+1
    os += -1/2*coeff, p01, 2*i-3, ppp, 2*i-1, p10, 2*i+1
    os += -1/2*coeff, p10, 2*i-3, ppp, 2*i-1, p01, 2*i+1
    os += -1/2*coeff, p11, 2*i-3, ppp, 2*i-1, p00, 2*i+1
    
    # Terms below add -1/2  I ⊗ (L_0^1^\dagger L_0^1)^T --> Operates only on R <=> even
    os += -1/2*coeff, p00, 2*(i-1), ppp, 2*i, p11, 2*(i+1)
    os += -1/2*coeff, p10, 2*(i-1), ppp, 2*i, p01, 2*(i+1)
    os += -1/2*coeff, p01, 2*(i-1), ppp, 2*i, p10, 2*(i+1)
    os += -1/2*coeff, p11, 2*(i-1), ppp, 2*i, p00, 2*(i+1)
    return os
end

function L02(i, proj=[p00, p01, p10, p11, ppp, pmm], coeff=1)
    os = OpSum()
    p00, p01, p10, p11, ppp, pmm = proj
    # Applies the opreator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os +=  coeff, p10, 2*i-3, p10, 2*i-2, ppp, 2*i-1, ppp, 2*i, p11, 2*i+1, p11, 2*i+2
    os += -coeff, p10, 2*i-3, p11, 2*i-2, ppp, 2*i-1, ppp, 2*i, p11, 2*i+1, p10, 2*i+2
    os += -coeff, p11, 2*i-3, p10, 2*i-2, ppp, 2*i-1, ppp, 2*i, p10, 2*i+1, p11, 2*i+2
    os +=  coeff, p11, 2*i-3, p11, 2*i-2, ppp, 2*i-1, ppp, 2*i, p10, 2*i+1, p10, 2*i+2
    
    # Terms below add -1/2 (L_0^2^\dagger L_0^2) ⊗ I --> Operates only on L <=> odd
    os += -1/2*coeff, p00, 2*i-3, ppp, 2*i-1, p11, 2*i+1
    os +=  1/2*coeff, p01, 2*i-3, ppp, 2*i-1, p10, 2*i+1
    os +=  1/2*coeff, p10, 2*i-3, ppp, 2*i-1, p01, 2*i+1
    os += -1/2*coeff, p11, 2*i-3, ppp, 2*i-1, p00, 2*i+1
    
    # Terms below add -1/2  I ⊗ (L_0^2^\dagger L_0^2)^T --> Operates only on R <=> even
    os += -1/2*coeff, p00, 2*(i-1), ppp, 2*i, p11, 2*(i+1)
    os +=  1/2*coeff, p10, 2*(i-1), ppp, 2*i, p01, 2*(i+1)
    os +=  1/2*coeff, p01, 2*(i-1), ppp, 2*i, p10, 2*(i+1)
    os += -1/2*coeff, p11, 2*(i-1), ppp, 2*i, p00, 2*(i+1)
    
    return os
end

function L03(i, proj=[p00, p01, p10, p11, ppp, pmm], coeff=1)
    os = OpSum()
    p00, p01, p10, p11, ppp, pmm = proj
    # Applies the opreator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os += coeff, p00, 2*i-3, p00, 2*i-2, pmm, 2*i-1, pmm, 2*i, p10, 2*i+1, p10, 2*i+2
    os += coeff, p00, 2*i-3, p01, 2*i-2, pmm, 2*i-1, pmm, 2*i, p10, 2*i+1, p11, 2*i+2
    os += coeff, p01, 2*i-3, p00, 2*i-2, pmm, 2*i-1, pmm, 2*i, p11, 2*i+1, p10, 2*i+2
    os += coeff, p01, 2*i-3, p01, 2*i-2, pmm, 2*i-1, pmm, 2*i, p11, 2*i+1, p11, 2*i+2
    
    # Terms below add -1/2 (L_0^3^\dagger L_0^3) ⊗ I --> Operates only on L <=> odd
    os += -1/2*coeff, p00, 2*i-3, pmm, 2*i-1, p00, 2*i+1
    os += -1/2*coeff, p01, 2*i-3, pmm, 2*i-1, p01, 2*i+1
    os += -1/2*coeff, p10, 2*i-3, pmm, 2*i-1, p10, 2*i+1
    os += -1/2*coeff, p11, 2*i-3, pmm, 2*i-1, p11, 2*i+1
    
    # Terms below add -1/2  I ⊗ (L_0^3^\dagger L_0^3)^T --> Operates only on R <=> even
    os += -1/2*coeff, p00, 2*(i-1), pmm, 2*i, p00, 2*(i+1)
    os += -1/2*coeff, p10, 2*(i-1), pmm, 2*i, p10, 2*(i+1)
    os += -1/2*coeff, p01, 2*(i-1), pmm, 2*i, p01, 2*(i+1)
    os += -1/2*coeff, p11, 2*(i-1), pmm, 2*i, p11, 2*(i+1)
    
    return os
end

function L04(i, proj=[p00, p01, p10, p11, ppp, pmm], coeff=1)
    os = OpSum()
    p00, p01, p10, p11, ppp, pmm = proj
    # Applies the opreator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os +=  coeff, p10, 2*i-3, p10, 2*i-2, pmm, 2*i-1, pmm, 2*i, p00, 2*i+1, p00, 2*i+2
    os += -coeff, p10, 2*i-3, p11, 2*i-2, pmm, 2*i-1, pmm, 2*i, p00, 2*i+1, p01, 2*i+2
    os += -coeff, p11, 2*i-3, p10, 2*i-2, pmm, 2*i-1, pmm, 2*i, p01, 2*i+1, p00, 2*i+2
    os +=  coeff, p11, 2*i-3, p11, 2*i-2, pmm, 2*i-1, pmm, 2*i, p01, 2*i+1, p01, 2*i+2
    
    # Terms below add -1/2 (L_0^4^\dagger L_0^4) ⊗ I --> Operates only on L <=> odd
    os += -1/2*coeff, p00, 2*i-3, pmm, 2*i-1, p00, 2*i+1
    os +=  1/2*coeff, p01, 2*i-3, pmm, 2*i-1, p01, 2*i+1
    os +=  1/2*coeff, p10, 2*i-3, pmm, 2*i-1, p10, 2*i+1
    os += -1/2*coeff, p11, 2*i-3, pmm, 2*i-1, p11, 2*i+1
    
    # Terms below add -1/2  I ⊗ (L_0^4^\dagger L_0^4)^T --> Operates only on R <=> even
    os += -1/2*coeff, p00, 2*(i-1), pmm, 2*i, p00, 2*(i+1)
    os +=  1/2*coeff, p10, 2*(i-1), pmm, 2*i, p10, 2*(i+1)
    os +=  1/2*coeff, p01, 2*(i-1), pmm, 2*i, p01, 2*(i+1)
    os += -1/2*coeff, p11, 2*(i-1), pmm, 2*i, p11, 2*(i+1)
    return os
end

# Convention for choosing sites:
# 1 , 2 , 3 , 4 , 5 , 6 , ... ,2N-1, 2N 
# 1L, 1R, 2L, 2R, 3L, 3R, ... , NL, NR

function L0_plus(i, coeff = 1)
    # Applies the operator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os = OpSum()
    # Simplified version
    os += - coeff/4,  Id, 2*i
    os += coeff/4, Z, 2*i-3, Z, 2*i-2, X, 2*i-1, X, 2*i, Z, 2*i+1, Z, 2*i+2
    return os
end

function L0_minus(i, coeff = 1)
    # Applies the operator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os = OpSum()
    # Simplified version
    os += coeff/4, X, 2*i+1, X, 2*i+2 # not a boundary term twist ()
    os += coeff/4, Z, 2*i-2,  X, 2*i,  X, 2*i+1, Y_tilde, 2*i+2 # twist (-1,1,1)
    os += coeff/4, Z, 2*i-3, X, 2*i-1, Y_tilde, 2*i+1, X, 2*i+2 # twist (-1,1,1)
    os += coeff/4, Z, 2*i-3, Z, 2*i-2, X, 2*i-1, X, 2*i, Y_tilde, 2*i+1, Y_tilde, 2*i+2 # twist (1,1,1)
    
    os += - coeff/2, Id, 2*i-1 # not a boundary term twist (1,1,1)
    os +=   coeff/4, Z, 2*i-3, X, 2*i-1, Z, 2*i+1 # twist (-1,1,1)
    os +=   coeff/4, Z, 2*(i-1), X, 2*i, Z, 2*(i+1) # twist (-1,1,1)

    return os
end

function L0_dual(i, coeff = 0)
    # Applies the operator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os = OpSum()
    # Simplified version
    ppm = Z*(Id-X)/2
    pmm = (Id-X)/2
    os += coeff, ppm, 2*i-1, ppm, 2*i, X, 2*i+1, X, 2*i+2, Z, 2*i+3, Z, 2*i+4
    os += -coeff/2, pmm, 2*i-1 
    os += -coeff/2, pmm, 2*i 
    return os
end

function L0_minus_left(i, coeff = 1)
    # Applies the operator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os = OpSum()
    # Simplified version
    os += coeff/4, X, 2*i-3, X, 2*i-2
    os += coeff/4, X, 2*i-3,  Y_tilde, 2*i-2,  X, 2*i, Z, 2*i+2
    os += coeff/4, Y_tilde, 2*i-3, X, 2*i-2, X, 2*i-1, Z, 2*i+1
    os += coeff/4, Y_tilde, 2*i-3, Y_tilde, 2*i-2, X, 2*i-1, X, 2*i, Z, 2*i+1, Z, 2*i+2
    
    os += - coeff/2, Id, 2*i-1
    os +=   coeff/4, Z, 2*i-3, X, 2*i-1, Z, 2*i+1
    os +=   coeff/4, Z, 2*(i-1), X, 2*i, Z, 2*(i+1)
    return os
end

function L0_minus_KW(i, coeff = 1)
    # Applies the operator on sites (i-1)L, (i-1)R, iL, iR, (i+1)L, (i+1)R
    # The same as 2i-3, 2i-2, 2i-1, 2i, 2i+1, 2i+2
    os = OpSum()
    # Simplified version
    os += coeff/4, X, 2*i+1, X, 2*i+2
    os += coeff/4, Z, 2*i-2,  X, 2*i,  X, 2*i+1, Y_tilde, 2*i+2
    os += coeff/4, Z, 2*i-3, X, 2*i-1, Y_tilde, 2*i+1, X, 2*i+2
    os += coeff/4, Z, 2*i-3, Z, 2*i-2, X, 2*i-1, X, 2*i, Y_tilde, 2*i+1, Y_tilde, 2*i+2
    
    os += - coeff/2, Id, 2*i-1
    os +=   coeff/4, Z, 2*i-3, X, 2*i-1, Z, 2*i+1
    os +=   coeff/4, Z, 2*(i-1), X, 2*i, Z, 2*(i+1)

    return os
end

function L1(i, coeff=1)
    os = OpSum()
    os += coeff, Z, 2*i-1, Z, 2*i # twist (1,1,1)
    os += -coeff, Id, 2*i-1 # twist (1,1,1)
    return os
end

function L1_dual(i, coeff=1)
    return L1(i, coeff)
end

function L2(i, proj, coeff=1)

    p00, p01, p10, p11, ppp, pmm = proj
    os = OpSum()
    os += coeff, Z, 2*i-3, Z, 2*i-2, p01, 2*i-1, p01, 2*i, Z, 2*i+1, Z, 2*i+2 # compicated under strong symmetry boundary
    os += coeff, Z, 2*i-3, Z, 2*i-2, p10, 2*i-1, p10, 2*i, Z, 2*i+1, Z, 2*i+2 # Compicated under strong symetry boundary
    os += -coeff, Id, 2*i-1
    return os
end

function L2_simple(i, proj, coeff=1)

    p00, p01, p10, p11, ppp, pmm = proj
    os = OpSum()
    os += coeff, Z, 2*i-3, Z, 2*i-2, p01, 2*i-1, p01, 2*i, Z, 2*i+1, Z, 2*i+2
    os += -coeff/2, Id, 2*i-1
    return os
end

function L2_paper(i, proj, coeff=1)
    p00, p01, p10, p11, ppp, pmm = proj
    os = OpSum()
    os += coeff, Z, 2*i-3, Z, 2*i-2, X, 2*i-1, X, 2*i, Z, 2*i+1, Z, 2*i+2 
    os += -coeff, Id, 2*i-1
    return os
end

function L2_paper_dual(i, proj, coeff=0)
    return Lx(i, coeff)
end

function L3(i, proj,sign, coeff=1)
    p00, p01, p10, p11, ppp, pmm = proj
    os = OpSum()
    os += coeff/4, Z, 2*i-3, Z, 2*i-2, X*(Id .+ sign .* Z), 2*i-1, X*(Id .+ sign .* Z), 2*i, Z, 2*i+1, Z, 2*i+2
    os += -1/4*coeff, (Id + sign*Z), 2*i-1
    os += -1/4*coeff, (Id + sign*Z), 2*i
    return os
end

function Lx(i,coeff = 0)
    os = OpSum()
    os += coeff, X, 2*i-1, X, 2*i
    os += -coeff, Id, 2*i-1
    return os
end

function Lz(i,coeff = 0)
    os = OpSum()
    os += coeff, Z, 2*i-1, Z, 2*i
    os += -coeff, Id, 2*i-1
    return os
end

function Lzz(i,coeff = 0)
    os = OpSum()
    os += coeff, Z, 2*i-1, Z, 2*i, Z, 2i+3, Z, 2i+4
    os += -coeff, Id, 2*i-1
    return os
end

function Ly_tilde(i,coeff = 0)
    os = OpSum()
    os += coeff, Y_tilde, 2*i-1, Y_tilde, 2*i
    os += -coeff, Id, 2*i-1
    return os
end

function Lxx_consecutive(i,coeff = 0)
    os = OpSum()
    os += coeff, X, 2*i-1, X, 2*i, X, 2i+1, X, 2i+2
    os += -coeff, Id, 2*i-1
    return os
end

function L_one_site(i, A=X, coeff=0)
    # Has jump operators A_i 
    os = OpSum()
    os += coeff, A, 2i-1, conj.(A), 2i
    os += -coeff/2, (A'*A), 2i-1
    os += -coeff/2, conj.(A'*A), 2i
    return os
end

function L_two_site(i, A=X, B=X, coeff=0)
    # Has jump operators A_i B_{i+2}
    os = OpSum()
    os += coeff, A, 2i-1, conj.(A), 2i, B, 2i+3, conj.(B), 2i+4
    os += -coeff/2, (A'*A), 2i-1, (B'*B), 2i+3
    os += -coeff/2, conj.(A'*A), 2i, conj.(B'*B), 2i+4
    return os
end

function L_three_site(i, A=Z, B=X,C=Z,  coeff=0)
    # Has jump operators A_i B_{i+1} C_{i+2}
    os = OpSum()
    os += coeff, A, 2i-1, conj.(A), 2i, B, 2i+1, conj.(B), 2i+2, C, 2i+3, conj.(C), 2i+4
    os += -coeff/2, (A'*A), 2i-1, (B'*B), 2i+1, (C'*C), 2i+3
    os += -coeff/2, conj.(A'*A), 2i, conj.(B'*B), 2i+2, conj.(C'*C), 2i+4
    return os
end

function L_four_site(i, A=Z, B=X, C=X, D=Z, coeff=0)
    # Has jump operators A_i B_{i+1} C_{i+2} D_{i+3}
    os = OpSum()
    os += coeff, A, 2i-1, conj.(A), 2i, B, 2i+1, conj.(B), 2i+2, C, 2i+3, conj.(C), 2i+4, D, 2i+5, conj.(D), 2i+6
    os += -coeff/2, (A'*A), 2i-1, (B'*B), 2i+1, (C'*C), 2i+3, (D'*D), 2i+5
    os += -coeff/2, conj.(A'*A), 2i, conj.(B'*B), 2i+2, conj.(C'*C), 2i+4, conj.(D'*D), 2i+6
    return os
end

function L_weak_1(i, proj, coeff=1)
    p00, p01, p10, p11, ppp, pmm = proj
    os = OpSum()
    os += coeff, p00,  2i-1, p00, 2i, p01, 2i+3, p01, 2i+4
    os += coeff, p00,  2i-1, p11, 2i, p01, 2i+3, p10, 2i+4
    os += coeff, p11,  2i-1, p00, 2i, p10, 2i+3, p01, 2i+4
    os += coeff, p11,  2i-1, p11, 2i, p10, 2i+3, p10, 2i+4

    os += -1/2*coeff, p00, 2i-1, p11, 2i+3
    os += -1/2*coeff, p11, 2i-1, p00, 2i+3

    os += -1/2*coeff, p00, 2i, p11, 2i+4
    os += -1/2*coeff, p11, 2i, p00, 2i+4
    return os
end

function Lbc_weak(N, CL0, CL1, CL2, proj)
    p00, p01, p10, p11, ppp, pmm = proj
    os = OpSum()
    coeff = CL0
    # Simplified version
    os +=  coeff/4, X, 1, X, 2 
    os += -coeff/4, Z, 2*N-2, X, 2*N,  X, 1, Y_tilde, 2
    os += -coeff/4, Z, 2*N-3, X, 2*N-1, Y_tilde, 1, X, 2 
    os +=  coeff/4, Z, 2*N-3, Z, 2*N-2, X, 2*N-1, X, 2*N, Y_tilde, 1, Y_tilde, 2
    
    os += - coeff/2, Id, 2*N-1 
    os += -  coeff/4, Z, 2*N-3, X, 2*N-1, Z, 1 
    os += -  coeff/4, Z, 2*(N-1), X, 2*N, Z, 2 

    p00, p01, p10, p11, ppp, pmm = proj
    coeff = CL2
    os += coeff, Z, 2N-1, Z, 2N, p01, 1, p01, 2, Z, 3, Z, 4 
    os += coeff, Z, 2N-1, Z, 2N, p10, 1, p10, 2, Z, 3, Z, 4 # Compicated under strong symetry boundary
    os += -coeff, Id, 1

    return os
end

function Lpbc(N, CL0, CL1, CL2, proj)
    os = OpSum()
    # Simplified version
    coeff = CL0
    os += coeff/4, X, 1, X, 2
    os += coeff/4, Z, 2*N-2,  X, 2*N,  X, 1, Y_tilde, 2
    os += coeff/4, Z, 2*N-3, X, 2*N-1, Y_tilde, 1, X, 2
    os += coeff/4, Z, 2*N-3, Z, 2*N-2, X, 2*N-1, X, 2N, Y_tilde, 1, Y_tilde, 2
    
    os += - coeff/2, Id, 2*N-1
    os +=   coeff/4, Z, 2*N-3, X, 2*N-1, Z, 1
    os +=   coeff/4, Z, 2*(N-1), X, 2*N, Z, 2*(1)

    p00, p01, p10, p11, ppp, pmm = proj
    coeff = CL2
    os += coeff, Z, 2N-1, Z, 2N, p01, 1, p01, 2, Z, 3, Z, 4 
    os += coeff, Z, 2N-1, Z, 2N, p10, 1, p10, 2, Z, 3, Z, 4 # Compicated under strong symetry boundary
    os += -coeff, Id, 1

    return os
end


function Lbc_strong(N, CL0, CL1, CL2, proj)
    os = OpSum()
    # Simplified version
    coeff = CL0
    os += coeff/4, X, 1, X, 2
    os += coeff/4, Z, 2*N-2,  X, 2*N,  X, 1, Y_tilde, 2
    os += coeff/4, Z, 2*N-3, X, 2*N-1, Y_tilde, 1, X, 2
    os += coeff/4, Z, 2*N-3, Z, 2*N-2, X, 2*N-1, X, 2N, Y_tilde, 1, Y_tilde, 2 
    
    os += - coeff/2, Id, 2*N-1
    os +=   coeff/4, Z, 2*N-3, X, 2*N-1, Z, 1
    os +=   coeff/4, Z, 2*(N-1), X, 2*N, Z, 2*(1)

    p00, p01, p10, p11, ppp, pmm = proj
    coeff = CL2
    os += -coeff, Z, 2N-1, Z, 2N, p01, 1, p01, 2, Z, 3, Z, 4 
    os += -coeff, Z, 2N-1, Z, 2N, p10, 1, p10, 2, Z, 3, Z, 4 # Compicated under strong symetry boundary
    os += -coeff, Id, 1

    return os
end



function Hx(i, CL_pert)
    os = OpSum()
    # Simplified version
    coeff = CL_pert
    os += -1im * coeff, X, 2i-1
    os += 1im * coeff, X, 2i
    return os
end

function Hz(i, CL_pert)
    os = OpSum()
    # Simplified version
    coeff = CL_pert
    os += -1im * coeff, Z, 2i-1
    os += 1im * coeff, Z, 2i
    return os
end

function Hzz(i, CL_pert)
    os = OpSum()
    # Simplified version
    coeff = CL_pert
    os += -1im * coeff, Z, 2i-1, Z, 2i+3
    os += 1im * coeff, Z, 2i, Z, 2i+4
    return os
end

function H_two_site(i, CL_pert, op1, op2)
    os = OpSum()

    coeff = CL_pert
    os += -1im * coeff, op1, 2i-1, op2, 2i+3
    os += 1im * coeff, conj.(op1)', 2i, conj.(op2)', 2i+4
    return os
end

function get_lind(sites, coeff = [1,0,0,0,0,0,1,1,0,0,0,0], pbc = false, bc_weak = false, bc_strong = false)
    # N is the Number of spins
    CL0, CL3, CLx_odd, CLx_even, CLz_odd, CLz_even, CL1, CL2, CL0_old, CLzz, CL0_left, CL0_dual, CL1_dual, CL2_dual, CL_pert = coeff
    # sites = siteinds("S=1/2",2*N)
    N = length(sites) ÷ 2
    os = OpSum()


    # println("WARNING: Using simplified Lindbladian")
    for i in 2:2:N-1
        # os += L0_plus(i, CL0) 
        os += L0_minus(i, CL0)
        os += L0_dual(i, CL0_dual)
        # os += L0_minus_left(i, CL0_left)
    end
    # for i in 3:2:N-1
    #     os += L3(i, proj,1, CL3) 
    #     os += L3(i, proj,-1, CL3)
    # end

    for i in 1:2:N
        os += L1(i, CL1)
        os += L1_dual(i, CL1_dual)
    end
    # println("Using L2 from paper")
    for i in 3:2:N-1
        os += L2_paper(i, proj, CL2) 
        os += L2_paper_dual(i, proj, CL2_dual) 
    end
    # for i in [N]
    #     os += Lx(i, CL_pert) 
    # end

    # for i in 1:2:N
    #     # os += Lx(i, CLx_odd) 
    #     os += L_one_site(i, p10, CL_pert)
    # end
    # for i in 2:2:N
    #     os += Lx(i, CLx_even) 
    # end
    # for i in 1:2:N
    #     os += Lz(i, CLz_odd) 
    # end
    # for i in 2:2:N
    #     os += Lz(i, CLz_even) 
    # end
    for i in 2:2:N-2
        os += Lzz(i, CL_pert) 
    end
    # for i in 2:2:N-1
    #     os += L01(i, proj, CL0_old) 
    #     os += L02(i, proj, CL0_old)
    #     os += L03(i, proj, CL0_old) 
    #     os += L04(i, proj, CL0_old)
    # end
    # for i in 1:2:N-2
    #     os += L_two_site(i, p00, p01, CL_pert)
    #     os += L_two_site(i, p11, p10, CL_pert)
    # end
    # for i in 2:2:N-2
    #     os += L_two_site(i, Z, Z, CL_pert)
    # end
    # for i in 2:2:N-2
    #     os += L_three_site(i,Z, X, Z, CL_pert)
    # end
    # for i in 2:2:N-3
    #     os += L_four_site(i, Z, Z, Y, Z, CL_pert)
    #     os += L_four_site(i, Z, X, X, Z, 1.5*CL_pert)
    # end
    # for i in 1:4:N-2
    #     os += L_weak_1(i, proj, CL_pert)
    # end
    # for i in 2:2:N-2
    #     os += L_two_site(i,pmp,pmp,  CL_pert)
    # end
    # for i in 1:2:N
    #     os += Hx(i, CL_pert)
    # end
    # for i in 2:2:N-2
    #     os += Hzz(i, CL_pert)
    # end
    # for i in 1:2:N-2
    #     os += H_two_site(i, CL_pert, X, X)
    # end

    if pbc == true
        println("Implementing PBC")
        os += Lpbc(N, CL0, CL1, CL2, proj)
    end
    if bc_weak == true
        println("Implementing Weak twisted BC")
        os += Lbc_weak(N, CL0, CL1, CL2, proj)
    end
    if bc_strong == true
        println("Implementing Strong twisted BC")
        os += Lbc_strong(N, CL0, CL1, CL2, proj)
    end

    # Make MPOs

    lind = MPO(os, sites);
    # lind_dag =  dag(swapprime(lind,0,1))
    # ldl = apply(lind_dag, lind);
    lind_dag = nothing 
    ldl = nothing
    println("Maximum bond dimension of Lindbladian = ", maximum(linkdims(lind)))
    # println("Maximum bond dimension of L^† L = ", maximum(linkdims(ldl)))
    return lind, ldl, lind_dag
end 

# ---------------------- Writing/Reading functions -------------------------
function saveMPS(filename, energy_super_list, state_super_list)
    f = h5open(filename,"w")
    for i in 1:length(state_super_list)
        for j in 1:length(state_super_list[i])
            write(f,"state_super_list"*string(i)*"_"*string(j), state_super_list[i][j])
            write(f,"energy_super_list"*string(i)*"_"*string(j), energy_super_list[i][j])
        end
    end
    close(f)
    return f
end

function save_singleMPS(filename, state_name, state)
    f = h5open(filename,"w")
    write(f,state_name, state)
    close(f)
    return f
end

function readMPS_folder(folder_name)
    filenames = readdir(folder_name; join=true)
    nmps = length(filenames) - 1
    slist = []
    for j in 1:nmps
        f = h5open(folder_name*string(j)*"_MPS.h5","r")
        psi = read(f,"psi",MPS)
        close(f)
        push!(slist, psi)
    end
    return slist
end

dagger(operator) = dag(swapprime(operator,0,1))

function get_V(state, symm)
    T1 = state[1]
    sites = siteinds(state)
    links = linkinds(state)
    T1_star = ITensor(conj.(Array(state[1], sites[1], links[1])), sites[1]', links[1]')
    U1 = ITensor(symm[1], sites[1], sites[1]')
    V1 = T1*U1*T1_star
    V_list = [V1]
    N = length(state)
    for i in 2:N-1
        Ti = state[i]
        Ti_star = ITensor(conj.(Array(state[i], sites[i], links[i-1], links[i])), sites[i]', links[i-1]', links[i]')
        Ui = ITensor(symm[i], sites[i], sites[i]')
        Vi = V_list[end] * Ui * Ti * Ti_star
        push!(V_list, Vi)
    end
    return V_list
end


function indicator_mixed(psi_1, j)
    N = length(psi_1)
    links = linkinds(psi_1)
    s = siteinds(psi_1)
    psi_dmrg = psi_1
    psi_dmrg = psi_dmrg/norm(psi_dmrg)

    orthogonalize!(psi_dmrg, N);

    oplist_weak = [i%4 == 1 || i%4 == 2 ? X : Id for i in 1:N]
    oplist_strong_left = [i%4 == 3 ? X : Id for i in 1:N]
    oplist_strong_right = [i%4 == 0 ? X : Id for i in 1:N]

    Vweak = get_V(psi_dmrg, oplist_weak)
    Vstrong_left = get_V(psi_dmrg, oplist_strong_left);
    Vstrong_right = get_V(psi_dmrg, oplist_strong_right);

    Vweak_matrix = Array(Vweak[j], inds(Vweak[j]))
    Vstrong_left_matrix = Array(Vstrong_left[j], inds(Vstrong_left[j]))
    Vstrong_right_matrix = Array(Vstrong_right[j], inds(Vstrong_right[j]))

    invariant_wsl = inv(Vweak_matrix) *inv(Vstrong_left_matrix) * Vweak_matrix  * Vstrong_left_matrix
    invariant_wsr = inv(Vweak_matrix) *inv(Vstrong_right_matrix) * Vweak_matrix  * Vstrong_right_matrix
    invariant_ss = inv(Vstrong_left_matrix) *inv(Vstrong_right_matrix) * Vstrong_left_matrix  * Vstrong_right_matrix

    return tr(invariant_wsl)/size(invariant_wsl)[1], tr(invariant_wsr)/size(invariant_wsr)[1] , tr(invariant_ss)/size(invariant_ss)[1] 
end

function weak_string_correlator(state::MPS, n::Int64, m::Int64)
    # Weak 1-copy string correlator. 
    # n and m are assumed to be even
    s = siteinds(state)
    ψ = get_ψid(s)
    denominator = inner(ψ, state)
    if abs(denominator) < 1e-5
        println("Warning from weak_string_correlator: trace too small")
    end
    l = linkinds(ψ)
    s = siteinds(ψ)
    for i in [2n-1, 2n, 2m-1, 2m]
        ψ[i] = ITensor(Z, l[i-1], l[i], s[i])
    end
    for i in 2n+1:4:2m-3
        ψ[i] = ITensor(X, l[i-1], l[i], s[i])
        ψ[i+1] = ITensor(X, l[i], l[i+1], s[i+1])
    end    
    return inner(ψ, state)/denominator
end

function weak_trivial_string_correlator(state::MPS, n::Int64, m::Int64)
    # Weak 1-copy string correlator. 
    # n and m are assumed to be even
    s = siteinds(state)
    ψ = get_ψid(s)
    denominator = inner(ψ, state)
    if abs(denominator) < 1e-5
        println("Warning from weak_string_correlator: trace too small")
    end
    l = linkinds(ψ)
    s = siteinds(ψ)
    # for i in [2n-1, 2n, 2m-1, 2m]
    #     ψ[i] = ITensor(Z, l[i-1], l[i], s[i])
    # end
    for i in 2n+1:4:2m-3
        ψ[i] = ITensor(X, l[i-1], l[i], s[i])
        ψ[i+1] = ITensor(X, l[i], l[i+1], s[i+1])
    end    
    return inner(ψ, state)/denominator
end

function string_correlator(state::MPS, n::Int64, m::Int64)
    # Strong 1 copy non-trivial string correlator
    s = siteinds(state)
    ψ = get_ψid(s)
    denominator = inner(ψ,state)
    if abs(denominator) < 1e-5
        println("Warning from string_correlator: trace too small")
    end
    l = linkinds(ψ)
    s = siteinds(ψ)
    for i in [2n, 2m]
        ψ[i] = ITensor(Z, l[i-1], l[i], s[i])
    end
    for i in 2n+2:4:2m-2
        ψ[i] = ITensor(X, l[i-1], l[i], s[i])
    end    
    return inner(ψ, state)/denominator
end

function sc5_one_string(ψ, n, m, normalized = false)
    # This is the strong two copy non-trivial string correlator
    
    state = deepcopy(ψ)
    s = siteinds(state)
    
    # swp = swapMPO(s)
    
    state = multiply(state, Z, 2n-1)
    # state = multiply(state, Z, 2n)
    state = multiply(state, Z, 2m-1)
    # state = multiply(state, Z, 2m)
    # println("using iY as end points")
    for i in 2n+1:4:2m-2
        state = multiply(state, X, i)
        # state = multiply(state, X, i+1)
    end
    denominator = 1.0
    if !normalized
        denominator = inner(ψ, ψ)
    end
    # denominator = inner(ψ', swp, ψ)
    if abs(denominator) < 1e-5
        println("Warning: denominator is very small = ", denominator)
    end

    return inner(ψ, state)/denominator
    # return inner(ψ', swp, state)/denominator
end

function trivial_correlator(state::MPS, n::Int64, m::Int64)
    # Strong 1 copy trivial string correlator
    s = siteinds(state)
    ψ = get_ψid(s)
    l = linkinds(ψ)
    s = siteinds(ψ)
    for i in 2n+2:4:2m-2
        ψ[i] = ITensor(X, l[i-1], l[i], s[i])
    end
    denominator = traceMPS(state)
    if abs(denominator) < 1e-5
        println("Warning from string_correlator: trace too small")
    end
    return inner(ψ, state)/denominator
end

function weak_nontrivial_2copy_correlator(ψ::MPS, n::Int64, m::Int64, normalized = false)
    # This is weak two copy non-trivial string correlator
    state = deepcopy(ψ)
    s = siteinds(state)
    if isodd(n) == true || isodd(m) == true
        println("n or m or both are odd")
        # return
    end
    # swp = swapMPO(s)
    state = multiply_tensor!(state, 2n-1, Z)
    state = multiply_tensor!(state, 2m-1, Z)
    state = multiply_tensor!(state, 2n, Z)
    state = multiply_tensor!(state, 2m, Z)
    for i in 2n+1:4:2m-2
        state = multiply_tensor!(state, i, X)
        state = multiply_tensor!(state, i+1, X)
    end
    denominator = 1.0
    if !normalized
        denominator = norm(ψ)^2
    end
    # denominator = inner(ψ', swp, ψ)
    if abs(denominator) < 1e-5
        println("Warning: denominator is very small = ", denominator)
    end

    return inner(ψ, state)/denominator
    # return inner(ψ', swp, state)/denominator
end

function weak_trivial_2copy_correlator(ψ::MPS, n::Int64, m::Int64, normalized = false)
    # This is weak two copy trivial string correlator
    state = deepcopy(ψ)
    s = siteinds(state)
    if isodd(n) == true || isodd(m) == true
        println("n or m or both are odd")
        return
    end
    # swp = swapMPO(s)
    
    for i in 2n+1:4:2m-2
        state = multiply_tensor!(state, i, X)
        state = multiply_tensor!(state, i+1, X)
    end
    denominator = 1.0
    if !normalized
        denominator = inner(ψ, ψ)
    end
    # denominator = inner(ψ', swp, ψ)
    if abs(denominator) < 1e-5
        println("Warning: denominator is very small = ", denominator)
    end

    return inner(ψ, state)/denominator
    # return inner(ψ', swp, state)/denominator
end

function strong_trivial_2copy_correlator(ψ::MPS, n::Int64, m::Int64, normalized = false)
    # This is strong two copy trivial string correlator
    state = deepcopy(ψ)
    s = siteinds(state)
    if iseven(n) == true || iseven(m) == true
        println("n or m or both are even")
        return
    end
    # swp = swapMPO(s)
    
    for i in 2n+1:4:2m-2
        state = multiply_tensor!(state,  i, X)
    end
    denominator = 1.0
    if !normalized
        denominator = inner(ψ, ψ)
    end
    # denominator = inner(ψ', swp, ψ)
    if abs(denominator) < 1e-5
        println("Warning: denominator is very small = ", denominator)
    end

    return inner(ψ, state)/denominator
    # return inner(ψ', swp, state)/denominator
end

function strong_nontrivial_2copy_correlator(ψ::MPS, n::Int64, m::Int64, normalized = false)
    # This is strong two copy non-trivial string correlator
    state = deepcopy(ψ)
    s = siteinds(state)
    if iseven(n) == true || iseven(m) == true
        println("n or m or both are even")
        return
    end
    # swp = swapMPO(s)

    state = multiply_tensor!(state, 2n-1, Z)
    state = multiply_tensor!(state, 2m-1, Z)
    for i in 2n+1:4:2m-2
        state = multiply_tensor!(state, i, X)
    end
    denominator = 1.0
    if !normalized
        denominator = inner(ψ, ψ)
    end
    # denominator = inner(ψ', swp, ψ)
    if abs(denominator) < 1e-5
        println("Warning: denominator is very small = ", denominator)
    end

    return inner(ψ, state)/denominator
    # return inner(ψ', swp, state)/denominator
end

function strong_nontrivial_1copy_correlator(state::MPS, n::Int64, m::Int64, denominator = false)
    # Strong 1 copy non-trivial string correlator
    # n, m should be odd
    s = siteinds(state)
    ψ = get_ψid(s)
    if denominator == false
        denominator = inner(ψ,state)
    end
    if iseven(n) == true || iseven(m) == true
        println("n or m or both are even")
        return
    end
    if abs(denominator) < 1e-5
        println("Warning from string_correlator: trace too small")
    end
    l = linkinds(ψ)
    s = siteinds(ψ)
    for i in [2n, 2m]
        ψ[i] = ITensor(Z, l[i-1], l[i], s[i])
    end
    for i in 2n+2:4:2m-2
        ψ[i] = ITensor(X, l[i-1], l[i], s[i])
    end    
    return inner(ψ, state)/denominator
end

function string_correlator(state::MPS, n::Int64, m::Int64)
    # Strong 1 copy non-trivial string correlator
    s = siteinds(state)
    ψ = get_ψid(s)
    denominator = inner(ψ,state)
    if abs(denominator) < 1e-5
        println("Warning from string_correlator: trace too small")
    end
    l = linkinds(ψ)
    s = siteinds(ψ)
    for i in [2n, 2m]
        ψ[i] = ITensor(Z, l[i-1], l[i], s[i])
    end
    for i in 2n+2:4:2m-2
        ψ[i] = ITensor(X, l[i-1], l[i], s[i])
    end    
    return inner(ψ, state)/denominator
end



function strong_trivial_1copy_correlator(ψ, n, m, denominator = false)
    # This is strong one copy trivial string correlator
    state = deepcopy(ψ)
    s = siteinds(state)
    if iseven(n) == true || iseven(m) == true
        println("n or m or both are even")
        return
    end
    ψ = get_ψid(s)
    if denominator == false
        denominator = inner(ψ,state)
    end
    if abs(denominator) < 1e-5
        println("Warning from strong_nontrivial_1copy_correlator: trace too small")
    end
    l = linkinds(ψ)
    s = siteinds(ψ)
    # for i in [2n, 2m]
    #     ψ[i] = ITensor(Z, l[i-1], l[i], s[i])
    # end
    for i in 2n+2:4:2m-2
        ψ[i] = ITensor(X, l[i-1], l[i], s[i])
    end    
    return inner(ψ, state)/denominator
    # return inner(ψ', swp, state)/denominator
end

# -----------------------------------------------------------------
# ----------------------- SSB functions ---------------------------
# -----------------------------------------------------------------

function multiply_tensor!(psi, i, op)
    s = siteinds(psi)
    O1 = ITensor(op, s[i],s[i]' )
    new_n = psi[i] * O1
    noprime!(new_n)
    psi[i] = new_n
    return psi
end

function ZZ_corr_1copy(psi::MPS, n::Int64, m::Int64, op=Z)
    
    ψid = get_ψid(siteinds(psi))
    den = inner(ψid, psi)
    if abs(den) < 1e-5
        println("denominator is small")
    end

    phi = deepcopy(psi)
    phi = multiply_tensor!(phi, 2n-1, op)
    phi = multiply_tensor!(phi, 2m-1, op)
    exp12 = inner(ψid, phi)/den

    phi = deepcopy(psi)
    phi = multiply_tensor!(phi, 2n-1, op)
    exp1 = inner(ψid, phi)/den

    phi = deepcopy(psi)
    phi = multiply_tensor!(phi, 2m-1, op)
    exp2 = inner(ψid, phi)/den
    
    return (exp12-exp1*exp2)
end

function ZZ_corr(psi::MPS, n::Int64, m::Int64, op=Z)
    nrm2 = norm(psi)^2

    phi = deepcopy(psi)
    phi = multiply_tensor!(phi, 2n-1, op)
    phi = multiply_tensor!(phi, 2m-1, op)
    exp12 = inner(phi, psi)/nrm2

    phi = deepcopy(psi)
    phi = multiply_tensor!(phi, 2n-1, op)
    exp1 = inner(phi, psi)/nrm2

    phi = deepcopy(psi)
    phi = multiply_tensor!(phi, 2m-1, op)
    exp2 = inner(phi, psi)/nrm2
    
    return (exp12-exp1*exp2)
end


function ZZZZ_corr(psi::MPS, n::Int64, m::Int64)
    nrm2 = norm(psi)^2

    phi = deepcopy(psi)
    
    phi = multiply_tensor!(phi, 2n-1, Z)
    phi = multiply_tensor!(phi, 2n, Z)
    expn = inner(phi, psi)/nrm2

    phi = multiply_tensor!(phi, 2m-1, Z)
    phi = multiply_tensor!(phi, 2m, Z)
    exp2 = inner(phi, psi)/nrm2

    phi = deepcopy(psi)
    phi = multiply_tensor!(phi, 2m-1, Z)
    phi = multiply_tensor!(phi, 2m, Z)
    expm = inner(phi, psi)/nrm2

    # return abs(exp2)
    return (exp2 - expn*expm)
end

function ZZZZ_corr2(psi::MPS, n::Int64, m::Int64)
    phi = psi/norm(psi)
    s = siteinds(phi)
    
    os = OpSum()
    os += 1.0, Z, 2n-1, Z, 2n, Z, 2m-1, Z, 2m
    op1 = MPO(os, s)

    os = OpSum()
    os += 1.0, Z, 2n-1, Z, 2n
    opn = MPO(os, s)

    os = OpSum()
    os += 1.0, Z, 2m-1, Z, 2m
    opm = MPO(os, s)

    return ( inner(phi', op1, phi) - inner(phi', opn, phi) * inner(phi', opm, phi))
end

function exp_val(state, i, op, normalized = false)
    state2 = deepcopy(state)
    state2 = multiply_tensor!(state2, i, op)
    if normalized
        return inner(state, state2)
    else
        return inner(state, state2)/norm(state)^2
    end
end

function exp_val_generalized(state, i, op, normalized = false)
    state2 = deepcopy(state)
    # op is a list of operators
    # op1_i, op2_{i+1} ... opn_{i+n-1}
    for j in 1:length(op)
        state2 = multiply_tensor!(state2, i+j-1, op[j])
    end
    if normalized
        return inner(state, state2)
    else
        return inner(state, state2)/norm(state)^2
    end
end


function mean_op(state, op=X)
    denominator = norm(state)^2
    sum1 = 0.0 + 0.0im
    for i in 3:4:length(state)
        sum1 += exp_val(state, i, op, true)/denominator
    end
    return sum1/(length(state)÷4)
end

# ---------------------------------------------------------
#  Perturbation theory Cijs
# ---------------------------------------------------------


function change_sites(ρ, new_sites)
    psi = deepcopy(ρ)
    for i in 1:length(new_sites)
        psi[i] = psi[i] * Id_tensor(new_sites[i], siteinds(psi)[i])
    end
    return psi
end

function cij_slow(ψ::MPS,i::Int64, j::Int64,  ρma_tilde::MPS)
    # ψ =ψ/traceMPS(ψ)
    # ρma = ρma/traceMPS(ρma)
    new_state = apply_op(ψ, Z, 2i-1)
    new_state = apply_op(new_state, Z, 2i)
    new_state = apply_op(new_state, Z, 2j-1)
    new_state = apply_op(new_state, Z, 2j)
    # swp = swapMPO(siteinds(ψ))
    
    # numerator = inner(ρma_tilde', swp, new_state)
    numerator = inner(ρma_tilde, new_state)
    # denominator = inner(ρma_tilde, ψ)
    # denominator = inner(ρma_tilde', swp, ψ)
    return numerator
end

function apply_Z!(ψ::MPS, s::Vector{Index{Int64}}, i::Int64)
    op_tensor = ITensor(Z, s[i], s[i]')
    newA = op_tensor * ψ[i]
    noprime!(newA)
    ψ[i] = newA
    return ψ
end

function cij_fast(ψ::MPS,i::Int64, j::Int64,  ρma_tilde::MPS)
    state = deepcopy(ρma_tilde)
    s = siteinds(state)
    state = apply_Z!(state, s, 2i-1)
    state = apply_Z!(state, s, 2i)
    state = apply_Z!(state, s, 2j-1)
    state = apply_Z!(state, s, 2j)
    return inner(ψ, state)
end

function αi(ψ::MPS,i::Int64, ρma_tilde::MPS)
    state = deepcopy(ρma_tilde)
    s = siteinds(state)
    state = apply_Z!(state, s, 2i)
    state = apply_Z!(state, s, 2(i+2))
    return inner(ψ, state)
end

function generate_cij(state1::MPS, state2::MPS)
    # state1 is ρma_tilde
    psi = plusplus_project(state2)
    cij_mat = fill(0.0+0.0im, length(psi)÷4,length(psi)÷4)
    psi = psi/norm(psi)
    ρma_tilde = plusplus_project(state1)
    ρma_tilde = change_sites(ρma_tilde, siteinds(state2))
    ρma_tilde = ρma_tilde / norm(ρma_tilde)
    s = siteinds(ρma_tilde)
    # temp2 =  []
    for i in 1:length(psi)÷4-1
        for j in i+1:length(psi)÷4
            state = deepcopy(ρma_tilde)
            state = apply_Z!(state, s, 4i-1)
            state = apply_Z!(state, s, 4i)
            state = apply_Z!(state, s, 4j-1)
            state = apply_Z!(state, s, 4j)
            state=state/norm(state)
            cij_mat[i,j] = inner(state, psi )
            # push!(temp2, cij_mat[i,j])
        end
    end
    return cij_mat, inner(ρma_tilde, psi)
end
