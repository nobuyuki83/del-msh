

def screened_poisson(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param: float, epsilon: float, stream_ptr=0):
    from ..del_msh_dlpack import nbody_screened_poisson

    nbody_screened_poisson(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param, epsilon, stream_ptr)


def elastic(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param: float, epsilon: float, stream_ptr=0):
    from ..del_msh_dlpack import nbody_elastic

    nbody_elastic(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param, epsilon, stream_ptr)