using MPIDistributedArrays
using MPI

function test()
    MPIDistributedArrays.initialize()
    nprocs = MPIDistributedArrays.Comm_size()
    
    comm = MPI.COMM_WORLD
    dims = (3,3,12)
    parallel = [1,1,nprocs]
    println("MArray")
    A_M =  MArray{Float64}(comm,dims,parallel,debug= true)
    A_M =  MArray(Float64,comm,dims,parallel,debug= true)
    A_M =  MArray{Float64}(dims,parallel,debug= true)
    A_M =  MArray(Float64,dims,parallel,debug= true)
    println("mzeros")
    A_M = mzeros(Float64,dims,parallel,debug=true)
    A_M = mzeros(dims,parallel,debug=true)
    A_M = mzeros(comm,dims,parallel,debug=true)
    A_M = mzeros(Float64,comm,dims,parallel,debug=true)
    println("distribute")
    A = rand(4,4,12)
    A_M = distribute(A,parallel,debug=true)
    println(A[1,1,1])

end
test()