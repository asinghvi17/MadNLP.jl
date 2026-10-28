# CPU SparseKKT image cache restoration; uses existing constructors and aliases.
    function restore_views(p::PrimalVector)
        PrimalVector(p.values,p.values_lr,p.values_ur,
            _madnlp_unsafe_wrap(p.values,length(p.x)),
            _madnlp_unsafe_wrap(p.values,length(p.s),length(p.x)+1))
    end
    function restore_views(p::UnreducedKKTVector)
        n,m,nl,nu=length(p.xp),length(p.xl),length(p.xzl),length(p.xzu)
        x=_madnlp_unsafe_wrap(p.values,n+m)
        xp=_madnlp_unsafe_wrap(p.values,n)
        UnreducedKKTVector(p.values,x,xp,view(xp,p.xp_lr.indices[1]),view(xp,p.xp_ur.indices[1]),
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
        s.x=restore_views(s.x);s.zl=restore_views(s.zl);s.zu=restore_views(s.zu)
        s.xl=restore_views(s.xl);s.xu=restore_views(s.xu);s.f=restore_views(s.f)
        s.x_trial=restore_views(s.x_trial)
        s.d=restore_views(s.d);s.p=restore_views(s.p)
        s._w1=restore_views(s._w1);s._w2=restore_views(s._w2)
        s._w3=restore_views(s._w3);s._w4=restore_views(s._w4)
        s.cnt.start_time=time() # image construction time is not runtime start time
        s.kkt=restore_views(s.kkt)
        it=s.iterator
        s.iterator=RichardsonIterator(s.kkt,it.opt,it.cnt,it.logger)
        s
    end
