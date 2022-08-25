using Copulas

# Note: hand writing bivariate copulas right now since Copulas.jl can have issues with points near the edge of the unit square

function bivarcdf(C::GumbelCopula, u₁::Real, u₂::Real)
    exp(-((-log(u₁))^C.θ + (- log(u₂))^C.θ)^(1.0/C.θ))
end

function bivarlogcdf(C::GumbelCopula, u₁::Real, u₂::Real)
    -((-log(u₁))^C.θ + (- log(u₂))^C.θ)^(1.0/C.θ)
end

function bivarpdf(C::GumbelCopula, u₁::Real, u₂::Real)
    logu₁ = log(u₁)
    logu₂ = log(u₂)
    logu₁C = (-logu₁)^C.θ
    logu₂C = (-logu₂)^C.θ
    bivarcdf(C,u₁,u₂)*(u₁*u₂)^(-1.0)* (logu₁C + logu₂C)^(-2.0 + 2.0/C.θ) * (logu₁*logu₂)^(C.θ-1.0) * (1.0 + (C.θ-1.0)*(logu₁C + logu₂C)^(-1.0/C.θ))
end

function bivarlogpdf(C::GumbelCopula, u₁::Real, u₂::Real)
    logu₁ = log(u₁)
    logu₂ = log(u₂)
    logu₁C = (-logu₁)^C.θ
    logu₂C = (-logu₂)^C.θ
    thetaminusone = (C.θ-1.0)
    bivarlogcdf(C,u₁,u₂) -1.0*log(u₁*u₂) + (-2.0 + 2.0/C.θ)*log(logu₁C + logu₂C) + thetaminusone*log(logu₁*logu₂) +log(1.0 + thetaminusone*(logu₁C + logu₂C)^(-1.0/C.θ))
end

function bivarcdf(C::ClaytonCopula, u₁::Real, u₂::Real)
    (u₁^(-C.θ) + u₂^(-C.θ)-1.0)^(-1.0/C.θ)
end

function bivarpdf(C::ClaytonCopula, u₁::Real, u₂::Real)
    (C.θ+1.0)*(u₁*u₂)^(-C.θ-1.0)*(u₁^(-C.θ)+u₂^(-C.θ)-1)^(-2-1.0/C.θ)
end

function bivarlogpdf(C::ClaytonCopula, u₁::Real, u₂::Real)
    log(C.θ+1.0)+ (-(C.θ+1.0))*log(u₁*u₂) + (-((2*C.θ+1)/C.θ))*log(u₁^(-C.θ)+u₂^(-C.θ)-1)
end

function bivarcdf(C::JoeCopula, u₁::Real, u₂::Real)
    oneminusu₁ = 1.0 - u₁
    oneminusu₂ = 1.0 - u₂
    1- (oneminusu₁^C.θ + oneminusu₂^C.θ - oneminusu₁^C.θ*oneminusu₂^C.θ)^(1/C.θ)
end

function bivarpdf(C::JoeCopula, u₁::Real, u₂::Real)
    oneminusu₁ = 1.0 - u₁
    oneminusu₂ = 1.0 - u₂
    thetaminusone = (C.θ-1.0)
    C.θ*(oneminusu₁^C.θ + oneminusu₂^C.θ - oneminusu₁^C.θ*oneminusu₂^C.θ)^(-1.0+1.0/C.θ) * oneminusu₁^thetaminusone * oneminusu₂^thetaminusone + thetaminusone*(oneminusu₁^C.θ + oneminusu₂^C.θ - oneminusu₁^C.θ* oneminusu₂^C.θ)^(-2.0+1.0/C.θ) * (1.0 - oneminusu₁^C.θ)*(1.0-oneminusu₂^C.θ)*oneminusu₁^thetaminusone*oneminusu₂^thetaminusone
end

function bivarlogpdf(C::JoeCopula, u₁::Real, u₂::Real)
    log(bivarpdf(C,u₁,u₂))
end

function bivarlogpdf(C::CT, u₁::Vector, u₂::Vector) where {CT<:Copulas.ArchimedeanCopula}
    @assert length(u₁) == length(u₂)
    return [bivarlogpdf(C,u₁[i],u₂[i]) for i in 1:length(u₁)]
end

function bivarpdf(C::CT, u₁::Vector, u₂::Vector) where {CT<:Copulas.ArchimedeanCopula}
    @assert length(u₁) == length(u₂)
    return [bivarpdf(C,u₁[i],u₂[i]) for i in 1:length(u₁)]
end