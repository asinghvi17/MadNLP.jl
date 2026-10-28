# CPU SparseKKT image cache restoration; uses existing constructors and aliases.
    function restore_views(p::PrimalVector,ind_lb,ind_ub)
        PrimalVector(p.values,view(p.values,ind_lb),view(p.values,ind_ub),
            _madnlp_unsafe_wrap(p.values,length(p.x)),
            _madnlp_unsafe_wrap(p.values,length(p.s),length(p.x)+1))
    end
    function restore_views(p::UnreducedKKTVector,ind_lb,ind_ub)
        n,m,nl,nu=length(p.xp),length(p.xl),length(p.xzl),length(p.xzu)
        x=_madnlp_unsafe_wrap(p.values,n+m)
        xp=_madnlp_unsafe_wrap(p.values,n)
        UnreducedKKTVector(p.values,x,xp,view(xp,ind_lb),view(xp,ind_ub),
            _madnlp_unsafe_wrap(p.values,m,n+1),
            _madnlp_unsafe_wrap(p.values,nl,n+m+1),
            _madnlp_unsafe_wrap(p.values,nu,n+m+nl+1))
    end
    function restore_views(k::SparseKKTSystem)
        n,m,nh,nj=length(k.pr_diag),length(k.du_diag),length(k.hess),length(k.jac_callback)
        V=k.aug_raw.V
        pr=_madnlp_unsafe_wrap(V,n)
        du=_madnlp_unsafe_wrap(V,m,n+nh+length(k.jac)+1)
        h=_madnlp_unsafe_wrap(V,nh,n+1)
        j=_madnlp_unsafe_wrap(V,length(k.jac),n+nh+1)
        jc=_madnlp_unsafe_wrap(V,nj,n+nh+1)
        k.hess_raw.V=h;k.jac_raw.V=j
        SparseKKTSystem(h,jc,j,k.quasi_newton,k.reg,pr,du,k.l_diag,k.u_diag,k.l_lower,k.u_lower,
            k.aug_raw,k.aug_com,k.aug_csc_map,k.hess_raw,k.hess_com,k.hess_csc_map,
            k.jac_raw,k.jac_com,k.jac_csc_map,k.linear_solver,k.ind_ineq,k.ind_lb,k.ind_ub)
    end
    function restore_views!(s::MadNLPSolver)
        s.x=restore_views(s.x,s.cb.ind_lb,s.cb.ind_ub);s.zl=restore_views(s.zl,s.cb.ind_lb,s.cb.ind_ub);s.zu=restore_views(s.zu,s.cb.ind_lb,s.cb.ind_ub)
        s.xl=restore_views(s.xl,s.cb.ind_lb,s.cb.ind_ub);s.xu=restore_views(s.xu,s.cb.ind_lb,s.cb.ind_ub);s.f=restore_views(s.f,s.cb.ind_lb,s.cb.ind_ub)
        s.x_trial=restore_views(s.x_trial,s.cb.ind_lb,s.cb.ind_ub)
        s.d=restore_views(s.d,s.cb.ind_lb,s.cb.ind_ub);s.p=restore_views(s.p,s.cb.ind_lb,s.cb.ind_ub)
        s._w1=restore_views(s._w1,s.cb.ind_lb,s.cb.ind_ub);s._w2=restore_views(s._w2,s.cb.ind_lb,s.cb.ind_ub)
        s._w3=restore_views(s._w3,s.cb.ind_lb,s.cb.ind_ub);s._w4=restore_views(s._w4,s.cb.ind_lb,s.cb.ind_ub)
        cb=s.cb
        s.c_slk=view(s.c,cb.ind_ineq)
        s.x_lr=view(s.x.values,cb.ind_lb);s.x_ur=view(s.x.values,cb.ind_ub)
        s.xl_r=view(s.xl.values,cb.ind_lb);s.xu_r=view(s.xu.values,cb.ind_ub)
        s.zl_r=view(s.zl.values,cb.ind_lb);s.zu_r=view(s.zu.values,cb.ind_ub)
        s.x_trial_lr=view(s.x_trial.values,cb.ind_lb);s.x_trial_ur=view(s.x_trial.values,cb.ind_ub)
        s.dx_lr=view(s.d.xp,cb.ind_lb);s.dx_ur=view(s.d.xp,cb.ind_ub)
        s.cnt.start_time=time() # image construction time is not runtime start time
        s.kkt=restore_views(s.kkt)
        if s.kkt.linear_solver isa LDLSolver
            M=s.kkt.linear_solver
            M.tril_to_full_view=view(M.tril.nzval,M.tril_to_full_view.indices[1])
        end
        it=s.iterator
        s.iterator=RichardsonIterator(s.kkt,it.opt,it.cnt,it.logger)
        s
    end

# Setup-only: Serialization materializes SubArrays and discards their old indices.
# Rebuild the existing sparse mapping before baking the image, never inside solve!.
function restore_serialized_setup!(s::MadNLPSolver)
    cb=s.cb
    s.cb=typeof(cb)(s.nlp,(getfield(cb,i) for i in 2:fieldcount(typeof(cb)))...)
    M=s.kkt.linear_solver
    if M isa LDLSolver
        _,mapping=get_tril_to_full(M.tril)
        M.tril_to_full_view=mapping
    end
    restore_views!(s)
end
