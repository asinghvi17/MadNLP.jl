using Test, MadNLP, Serialization
@testset "noncontiguous image view restoration" begin
    lb=[1,4];ub=[2,5]
    p=MadNLP.PrimalVector(Vector{Float64},3,2,lb,ub)
    p.values .= 1:5
    io=IOBuffer();serialize(io,p);seekstart(io);q=deserialize(io)
    r=MadNLP.restore_views(q,lb,ub)
    r.values_lr[2]=99
    @test r.values[4]==99
    @test r.s[1]==99
    r.values_ur[2]=101
    @test r.values[5]==101
    @test r.s[2]==101
    d=MadNLP.UnreducedKKTVector(Vector{Float64},5,2,2,2,lb,ub)
    io=IOBuffer();serialize(io,d);seekstart(io);e=MadNLP.restore_views(deserialize(io),lb,ub)
    e.xp_ur[2]=77
    @test e.values[5]==77
    e.xzl[1]=33
    @test e.values[8]==33
end
