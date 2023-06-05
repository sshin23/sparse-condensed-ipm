using PowerModels, MadNLP, MadNLPHSL, NLPModels, JuMP, SparseArrays, LinearAlgebra
PowerModels.silence()

pm = instantiate_model(
    # "/home/sshin/git/pglib-opf/pglib_opf_case30_ieee.m",
    # "/home/sshin/git/pglib-opf/pglib_opf_case300_ieee.m",
    "/home/sshin/git/pglib-opf/pglib_opf_case1354_pegase.m",
    # "/home/sshin/git/pglib-opf/pglib_opf_case2869_pegase.m",
    # "/home/sshin/git/pglib-opf/pglib_opf_case9241_pegase.m",
    # "/home/sshin/git/pglib-opf/pglib_opf_case2000_goc.m",
    ACPPowerModel,
    PowerModels.build_opf
)

set_optimizer(
    pm.model,
    () -> MadNLP.Optimizer(
        linear_solver = Ma57Solver
    )
)
optimize!(pm.model)

nlp = pm.model.moi_backend.optimizer.model.nlp

nlp.meta.ucon .+= 1e-8

tol = 1e-6

solver = MadNLPSolver(
    nlp;
    # linear_solver=Ma57Solver,
    linear_solver = MadNLPCUSOLVER.RFSolver
)

solve!(solver)
