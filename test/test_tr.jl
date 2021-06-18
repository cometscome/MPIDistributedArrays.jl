using MPIDistributedArrays
using MPI
MPIDistributedArrays.initialize()
comm = MPI.COMM_WORLD
myrank = MPI.Comm_rank(comm)
nprocs = MPIDistributedArrays.Comm_size()

function partial_tr(A::Array{T,3},NC,NV) where {T <: Number} 
    s = zero(T)
    for i=1:NV
        for k=1:NC
            s += A[k,k,i]
        end
    end
    return s
end


function loop_tr(A::MArray{T,N,B},NC,NV) where {T <: Number,N,B}
    #i_local = A.localindices
    s = zero(T)
    A2 = A.localpart
    #_,_,N3 = size(A2)
    #NC2 = NC + 2
    #NV2= A.localindices[3][end] - A.localindices[3][1] + 1# length(A.localindices[3])
    for i = 1:NV#length(A.localindices[3])
        for k=1:NC
            s += A2[k,k,i]
        end
    end
    return s
end

function partial_tr(A::MArray{T,N,B},NC,NV) where {T <: Number,N,B}
    A2 = A.localpart
    N3 = A.localsize[3]
    s = partial_tr(A2,NC,N3)
    return MPI.Allreduce(s,MPI.SUM,A.comm)
    #return reduce_sum(A,s)
end


function test()
    NC = 3
    NV = 256*1000

    NVs = 256*[1,10,100,200,500,1000,5000]
    sc = []
    pc = []
    for NV in NVs
        A_normal = rand(ComplexF64,NC,NC,NV)
        #num_workers = nworkers()


        n = 100
        ns = 10
        ks = []
        for k=1:n
            stime = time_ns()*10^(-3)
            tr_normal = partial_tr(A_normal,NC,NV)
            etime = time_ns()*10^(-3)
            push!(ks,(etime-stime))
            if myrank == 0
                #println("normal: $tr_normal")
                #println("$k-th ",(etime-stime)*10^6," micro sec")
            end
        end

        tr_normal = partial_tr(A_normal,NC,NV)
        if myrank == 0
            println(sum(ks[ns:end])/length(ks[ns:end])," micro sec")
            println("NV = $NV, normal: $tr_normal")
        end
        push!(sc,sum(ks[ns:end])/length(ks[ns:end]))

        MPI.Barrier(comm)
        if myrank == 0 
            println("MArray")
        end

        A_dist = distribute(A_normal,[1,1,nprocs])
        

        ks = []

        for k=1:n
            stime = time_ns()*10^(-3)
            #tr_dist = partial_tr(A_dist,NC,NV)
            tr_dist = partial_tr(A_dist,NC,NV)
            #tr_dist = partial_tr(A_normal,NC,NV)
            #tr_normal = partial_tr(A_normal,NC,NV)
            etime =time_ns()*10^(-3)
            push!(ks,(etime-stime))
            if myrank == 0
                #println("MArray: $tr_dist")
                #println("$k-th ",(etime-stime)*10^6," micro msec")
            end
        end

        tr_dist = partial_tr(A_dist,NC,NV)
        if myrank == 0
            println(sum(ks[ns:end])/length(ks[ns:end])," micro sec")
            println("NV = $NV, MArray: $tr_dist")
            #println("MArray: $tr_dist")
        end

        push!(pc,sum(ks[ns:end])/length(ks[ns:end]))
        free(A_dist )
    end

    if myrank == 0
        println(NVs)
        println(sc)
        println(pc)
    end

end
test()