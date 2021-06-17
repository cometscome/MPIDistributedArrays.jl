using MPIDistributedArrays
using Test
using MPI

testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "test_")
testfiles = sort(filter(istest, readdir(testdir)))

@testset "$f" for f in testfiles
#@testset "MPIDistributedArrays.jl" begin
    mpiexec() do cmd
        println("nproc = 1")
        run(`$cmd -n 1 $(Base.julia_cmd()) $(joinpath(testdir, f))`)
        @test true
        println("nproc = 2")
        run(`$cmd -n 2 $(Base.julia_cmd()) $(joinpath(testdir, f))`)
        @test true
    end
end
