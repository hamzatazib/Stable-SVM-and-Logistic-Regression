# Authors: Hamza Tazi Bouardi & Pierre-Henri Ramirez
using JuMP, Gurobi, CSV, LinearAlgebra, DataFrames, Random, Distributions, Statistics,MLBase, ROCAnalysis
gurobi_env = Gurobi.Env()

### Loading Data ###
train_data = CSV.read("data/adult_train.csv")
X_train = convert(Matrix, train_data[:, 1:91])
y_train = train_data[:, 92]
test_data = CSV.read("data/adult_test.csv")
X_test = convert(Matrix, test_data[:, 1:91])
y_test = test_data[:, 92]
println("Got the data for X dataset.")


### Utils Functions ###
function compute_∇f(w_k, z_k, y, X, λ)
    n, p = size(X)
    ∇f_k = sum(-z_k[i]/(1+exp(y[i]*dot(w_k,X[i,:])))*y[i].*X[i,:] for i in 1:n) + 2*λ.*w_k
    return ∇f_k
end

function solve_inner_max_problem(w_k, y, X, K, λ)
    n, p = size(X)
    model_inner_max = Model(solver=GurobiSolver(OutputFlag=0,gurobi_env))
    @variable(model_inner_max, z[1:n] >= 0)
    @constraint(model_inner_max, [i=1:n], 1 >= z[i])
    @constraint(model_inner_max, sum(z) <= K)
    @objective(
        model_inner_max,
        Max,
        sum(z[i]*log(1+exp(-y[i]*dot(X[i,:], w_k))) for i=1:n)
    )
    solve(model_inner_max)
    optimal_z_k = getvalue(z)
    optimal_f_k = getobjectivevalue(model_inner_max) + λ*dot(w_k,w_k)
    return optimal_z_k, optimal_f_k
end

function scores(preds, gt)
    acc = sum(preds .== gt)/size(preds)[1]
    TPR = dot((preds.==1),gt.==1)/(dot((preds.==1),gt.==1) + dot((preds.==-1),gt.==1))
    FPR = dot((preds.==1),gt.==-1)/ (dot((preds.==1),gt.==-1) + dot((preds.==-1),gt.==-1))
    return acc, TPR, FPR
end

### Cutting Planes Implementation ###
function stable_LR_cutting_planes(y, X, ε, K,λ)
    errors = []
    n, p = size(X)
    # Initialization values and step 0
    w_0 = [0 for i in 1:p]
    #w_0 = [rand(Uniform(-0.5, 0.5)) for i in 1:p]
    z_0, f_0 = solve_inner_max_problem(w_0, y, X, K, λ)
    ∇f_0 = compute_∇f(w_0, z_0, y, X, λ)

    # Outer minimization problem
    outer_min_model = Model(solver=GurobiSolver(OutputFlag=0, gurobi_env))
    @variable(outer_min_model, t >= 0)
    @variable(outer_min_model, w[1:p])
    #@constraint(outer_min_model, [j=1:p], -1 <= w[j] <= 1)
    @constraint(outer_min_model, t >= f_0 + dot(∇f_0, w)-dot(∇f_0, w_0))
    @constraint(outer_min_model, [j=1:p], 1 >= w[j])
    @constraint(outer_min_model, [j=1:p], w[j] >= -1)
    @objective(outer_min_model, Min, t)
    k = 1 # Number of constraints in the final problem
    solve(outer_min_model)

    # New steps k
    t_k = getvalue(t)
    w_k = getvalue(w)
    z_k, f_k = solve_inner_max_problem(w_k, y, X, K, λ)

    ∇f_k = compute_∇f(w_k, z_k, y, X, λ)
    while abs(f_k - t_k) >= ε # error
        push!(errors, f_k - t_k)
        @constraint(outer_min_model,t >= f_k + dot(∇f_k, w)-dot(∇f_k, w_k))
        k += 1
        solve(outer_min_model)
        # Updating all the values
        t_k = getvalue(t)
        w_k = getvalue(w)
        z_k, f_k = solve_inner_max_problem(w_k, y, X, K, λ)

        ∇f_k = compute_∇f(w_k, z_k, y, X, λ)
        if k%100 == 0
            println("Number of constraints: ", k, "\t Error = ", abs(t_k - f_k))
        end
        if k > 20000
            break
        end
    end
    push!(errors, f_k - t_k)
    return t_k, f_k, w_k, z_k, errors
end

