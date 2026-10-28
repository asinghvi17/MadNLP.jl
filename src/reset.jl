# Reset administrative state; all numerical work remains the stock initialize!/solve!.
# Keep the counters object: RichardsonIterator and execution stats alias it.
@generated function reset_counters!(c::MadNLPCounters)
    assigns=[:(setfield!(c,$(QuoteNode(n)),getfield(fresh,$(QuoteNode(n))))) for n in fieldnames(MadNLPCounters)]
    quote
        fresh=MadNLPCounters(start_time=time())
        $(assigns...)
        c
    end
end
function reset!(s::MadNLPSolver)
    reset_counters!(s.cnt)
    NLPModels.reset!(s.nlp)
    s.status=INITIAL
    s.RR=nothing
    empty!(s.filter)
    _reset!(s.kkt.quasi_newton)
    s.del_w=zero(s.del_w);s.del_w_last=zero(s.del_w_last);s.del_c=zero(s.del_c)
    s.alpha=zero(s.alpha);s.alpha_z=zero(s.alpha_z);s.ftype=" "
    s
end
