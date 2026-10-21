def eval_kelvinlet_bruteforce(
    tri2vtx, vtx2xyz, vtx2rhs, wtx2xyz, wtx2lhs, mu, nu, stream_ptr=0
):
    from ..del_msh_dlpack import trimesh3_bem_eval_kelvinlet_bruteforce

    trimesh3_bem_eval_kelvinlet_bruteforce(
        tri2vtx, vtx2xyz, vtx2rhs, wtx2xyz, wtx2lhs, mu, nu, stream_ptr
    )
