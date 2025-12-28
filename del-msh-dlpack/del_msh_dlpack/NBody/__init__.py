from dataclasses import dataclass
from typing import Literal, Union

@dataclass(frozen=True)
class ScreenedPoisson:
    lambda_: float
    eps: float


@dataclass(frozen=True)
class Elastic:
    nu: float
    eps: float

Model = Union[ScreenedPoisson, Elastic]


def filter_brute_force(vtx2co, vtx2rhs, wtx2co, wtx2lhs, model: Model, stream_ptr=0):

    match model:
        case ScreenedPoisson(lambda_=lambda_param, eps=epsilon):
            from ..del_msh_dlpack import nbody_screened_poisson

            nbody_screened_poisson(vtx2co, vtx2rhs, wtx2co, wtx2lhs, lambda_param, epsilon, stream_ptr)
        case Elastic(nu=nu, eps=epsilon):
            from ..del_msh_dlpack import nbody_elastic

            nbody_elastic(vtx2co, vtx2rhs, wtx2co, wtx2lhs, nu, epsilon, stream_ptr)


def filter_with_acceleration(
    vtx2co, vtx2rhs, wtx2co, wtx2lhs,
    model: Model,
    transform_world2unit,
    idx2jdx_offset, jdx2vtx,
    onodes, onode2center, onode2depth, onode2gcunit, onode2rhs,
    theta: float,
    stream_ptr):
    match model:
        case ScreenedPoisson(lambda_=lambda_param, eps=epsilon):
            from ..del_msh_dlpack import nbody_screened_poisson_with_acceleration

            nbody_screened_poisson_with_acceleration(
                vtx2co, vtx2rhs, wtx2co, wtx2lhs,
                lambda_param, epsilon,
                transform_world2unit,
                idx2jdx_offset, jdx2vtx,
                onodes, onode2center, onode2depth, onode2gcunit, onode2rhs,
                theta, stream_ptr)

        case Elastic(nu=nu, eps=epsilon):
            from ..del_msh_dlpack import nbody_elastic_with_acceleration

            nbody_elastic_with_acceleration(
                vtx2co, vtx2rhs, wtx2co, wtx2lhs,
                nu, epsilon,
                transform_world2unit,
                idx2jdx_offset, jdx2vtx,
                onodes, onode2center, onode2depth, onode2gcunit, onode2rhs,
                theta, stream_ptr)
