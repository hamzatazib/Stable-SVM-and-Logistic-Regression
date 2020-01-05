using JuMP, Gurobi, CSV, LinearAlgebra, DataFrames, Random, Distributions, Statistics
gurobi_env = Gurobi.Env()

### General Functions ###
function scores(y_pred, y_test)
    acc = sum(y_pred .== y_test)/size(y_pred)[1]
    TPR = dot((y_pred.==1),y_test.==1)/(dot((y_pred.==1),y_test.==1) + dot((y_pred.==-1),y_test.==1))
    FPR = dot((y_pred.==1),y_test.==-1)/ (dot((y_pred.==1),y_test.==-1) + dot((y_pred.==-1),y_test.==-1))
    return acc, TPR, FPR
end

function preprocess_y_multiclass(y, n, K)
    """
    This function takes y containing labels and returns y_matrix_binary
    which is a matrix of dimensions n×K for which y[i,k] = 1 if y_i = k, -1 otherwise
    """
    y_matrix_binary = zeros(n, K)
    for k=1:K
        for i=1:n
            if y[i] == k
                y_matrix_binary[i,k] = 1
            else
                y_matrix_binary[i,k] = -1
            end
        end
    end

    return y_matrix_binary

end


function fit_hyperplane_2_classes(x, w_opt, b_opt)
    value = x'w_opt + b_opt
    if value > 0
        return 1
    else
        return -1
    end
end



function fit_hyperplanes_K_classes(x, w_opt_matrix, b_opt_vect, K)
    values = zeros(K)
    for k=1:K
        values[k] = x'w_opt_matrix[:, k] + b_opt_vect[k]
    end
    return argmax(values) #number from 1 to K which will be the class !
end


function createTrainValidation(data_X, data_Y, train_rate)
    n, p = size(data_X)
    row_ids_shuffled = shuffle(1:n)
    train_ids = view(row_ids_shuffled, 1:floor(Int, train_rate*n))
    validation_ids = view(row_ids_shuffled, (floor(Int, train_rate*n)+1:n))
    return data_X[train_ids,:], data_X[validation_ids,:], data_Y[train_ids,1], data_Y[validation_ids,1]
end


function accuracy(y_test, y_pred)
    return round(length(findall(y_test .== y_pred))/length(y_test), digits=3)
end
##############################################################################################
##############################################################################################
### Model Implementations ###
# Classic Binary ℓ1 -norm Soft Margin SVM
function classic_svm_1_norm_binary(X, y, λ)
    # T: number of points in training, K: number of classes, λ: Regularization coeff
    n,d = size(X)
    svm_classic = Model(solver=GurobiSolver(OutputFlag = 0, gurobi_env))
    @variable(svm_classic, w[1:d])
    @variable(svm_classic, b)
    @variable(svm_classic, ξ[1:n] >= 0)

    @objective(
        svm_classic,
        Min,
        w'w + λ*sum(ξ[i] for i=1:n)
    )

    @constraint(
        svm_classic,
        [i=1:n],
        y[i]*(w'X[i,:] + b) >= 1 - ξ[i]
    )
    solve(svm_classic)
    w_opt_classic = getvalue(w)
    b_opt_classic = getvalue(b)
    obj_opt_classic = getobjectivevalue(svm_classic)

    return w_opt_classic, b_opt_classic, obj_opt_classic
end

# Stable Binary ℓ1 -norm Soft Margin SVM (Dualized formulation)
function stable_svm_1_norm_binary(X, y, λ, T)
    # T: number of points in training, K: number of classes, λ: Regularization coeff
    n,d = size(X)
    model_svm = Model(solver=GurobiSolver(OutputFlag = 0, gurobi_env))
    @variable(model_svm, w[1:d])
    @variable(model_svm, b)
    @variable(model_svm, θ)
    @variable(model_svm, u[1:n] >= 0)

    @objective(
        model_svm,
        Min,
        w'w + λ*(T*θ + sum(u[i] for i=1:n))
    )

    @constraint(model_svm, [i=1:n], u[i] + θ >= 0)
    @constraint(
        model_svm,
        [i=1:n],
        u[i] + θ >= 1-y[i]*(w'X[i,:] + b)
    )
    solve(model_svm)
    w_opt = getvalue(w)
    b_opt = getvalue(b)
    u_opt = getvalue(u)
    θ_opt = getvalue(θ)
    opt_obj = getobjectivevalue(model_svm)

    return w_opt, b_opt, u_opt, θ_opt, opt_obj
end

# Classic MultiClass ℓ1 -norm Soft Margin SVM
function classic_multiclass_svm_1_norm_multiclass(X, y, λ, K)
    # y must be preprocessed with the function in general functions
    # T: number of points in training, K: number of classes, λ: Regularization coeff
    n,d = size(X)
    svm_classic_multiclass = Model(solver=GurobiSolver(OutputFlag = 0, gurobi_env))
    @variable(svm_classic_multiclass, w[1:d, 1:K])
    @variable(svm_classic_multiclass, b[1:K])
    @variable(svm_classic_multiclass, ξ[1:n, 1:K] >= 0)

    @objective(
        svm_classic_multiclass,
        Min,
        sum(w[:,k]'w[:,k] for k=1:K) + λ*sum(sum(ξ[i, k] for i=1:n) for k=1:K)
    )

    @constraint(
        svm_classic_multiclass,
        [i=1:n, k=1:K],
        y[i,k]*(w[:,k]'X[i,:] + b[k]) >= 1 - ξ[i, k]
    )
    solve(svm_classic_multiclass)
    w_opt_classic_multiclass = getvalue(w)
    b_opt_classic_multiclass = getvalue(b)
    obj_opt_classic_multiclass = getobjectivevalue(svm_classic_multiclass)

    return w_opt_classic_multiclass, b_opt_classic_multiclass, obj_opt_classic_multiclass
end

# Stable MultiClass  ℓ1 -norm Soft Margin SVM
# This is a different implementation because in the Binary case,
# y_i  = +-1 but in the multiclass, all values of y_i are in [1,K]
function stable_svm_1_norm_multiclass(X, y, λ, T, K)
    # y must be preprocessed with the function in general functions
    # T: number of points in training, K: number of classes, λ: Regularization coeff
    n,d = size(X)
    model_svm = Model(solver=GurobiSolver(OutputFlag = 0, gurobi_env))
    @variable(model_svm, w[1:d, 1:K])
    @variable(model_svm, b[1:K])
    @variable(model_svm, θ)
    @variable(model_svm, η[1:n, 1:K] >= 0)
    @variable(model_svm, u[1:n] >= 0)

    @objective(
        model_svm,
        Min,
        sum(w[:,k]'w[:,k] for k=1:K) + λ*(T*θ + sum(u[i] for i=1:n))
    )

    @constraint(model_svm, [i=1:n], u[i] + θ >= sum(η[i,k] for k=1:K))
    @constraint(
        model_svm,
        [i=1:n, k=1:K],
        η[i,k] >= (1-y[i,k]*(w[:, k]'X[i,:] + b[k]))
    )
    solve(model_svm)
    w_opt = getvalue(w)
    b_opt = getvalue(b)
    opt_obj = getobjectivevalue(model_svm)

    return w_opt, b_opt, opt_obj
end