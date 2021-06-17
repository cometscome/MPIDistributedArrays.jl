using MPIDistributedArrays
using MPI
MPIDistributedArrays.initialize()
comm = MPI.COMM_WORLD
myrank = MPI.Comm_rank(comm)

function test()
    
    nprocs = MPIDistributedArrays.Comm_size()
    
    
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
    A = rand(2,2,4)
    if myrank == 0
        display(A)
        println("\t")
    end
    A_M = distribute(A,parallel,debug=true)
    println("myrank $myrank ",A_M[1,1,1])
    if myrank == 0
        display(A_M)
        println("\t")
    end
    MPI.Barrier(comm) 
    if myrank == 1
        println("\t")
        display(A_M)
        println("\t")
    end
    free(A_M)

end
test()