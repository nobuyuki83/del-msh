

def screened_poisson(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param: float, epsilon: float, stream_ptr=0):
    from ..del_msh_dlpack import nbody_screened_poisson

    nbody_screened_poisson(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param, epsilon, stream_ptr)


def elastic(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param: float, epsilon: float, stream_ptr=0):
    from ..del_msh_dlpack import nbody_elastic

    nbody_elastic(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param, epsilon, stream_ptr)


def screened_poisson_with_acceleration(
    vtx2co, vtx2rhs, wtx2co, wtx2lhs,
    lambda_, epsilon,
    transform_world2unit,
    idx2jdx_offset, jdx2vtx,
    onodes, onode2center, onode2depth, onode2gcunit, onode2rhs,
    theta,
    stream_ptr):
    from ..del_msh_dlpack import nbody_screened_poisson_with_acceleration

    nbody_screened_poisson_with_acceleration(
        vtx2co, vtx2rhs, wtx2co, wtx2lhs,
        lambda_, epsilon,
        transform_world2unit,
        idx2jdx_offset, jdx2vtx,
        onodes, onode2center, onode2depth, onode2gcunit, onode2rhs, theta, stream_ptr)