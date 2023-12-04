struct InteractionMatrix
    m11::Float64
    m12::Float64
    m13::Float64
    m21::Float64
    m22::Float64
    m23::Float64
    m31::Float64
    m32::Float64
    m33::Float64
end

function InteractionMatrix() # just for initialization
    return InteractionMatrix(zeros(Float64,9)...)
end

function InteractionMatrix(M::T) where T<:AbstractMatrix
    size(M) == (3,3) || error("Interaction matrix must be of size 3x3")
    m = InteractionMatrix(M[1,1], M[1,2], M[1,3], M[2,1], M[2,2], M[2,3], M[3,1], M[3,2], M[3,3])
    return m
end

# add a new function to modulate the coupling strength
function coefMatrix(M::InteractionMatrix, coef::Float64)
    cm = InteractionMatrix(coef*M.m11, coef*M.m12, coef*M.m13,
                           coef*M.m21, coef*M.m22, coef*M.m23,
                           coef*M.m31, coef*M.m32, coef*M.m33)
    return cm
end

# add a isotropic coefMatrix to the original matrix
function addMatrix(M::InteractionMatrix, coef::Float64)
    cm = InteractionMatrix(coef+M.m11,      M.m12,      M.m13,
                                M.m21, coef+M.m22,      M.m23,
                                M.m31,      M.m32, coef+M.m33)
    return cm
end