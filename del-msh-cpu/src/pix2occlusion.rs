use crate::trimesh3_raycast::ScalarRender;
pub struct Occlusion;

impl<T> ScalarRender<T> for Occlusion
where T: num_traits::Float
{
    fn fwd(&self, _: &[T; 3], i_tri: u32, _: &[u32], _: &[T], _: &[T; 16]) -> T {
        if i_tri == u32::MAX {
            T::zero()
        } else {
            T::one()
        }
    }

    fn bwd(
        &self,
        _: T,
        _: &[T; 3],
        _: &[T; 3],
        _: &[T; 3],
        _: &[T; 3],
        _: &[T; 3],
        _: &[T; 16],
    ) -> ([T; 3], [T; 3], [T; 3]) {
        let zero = T::zero();
        ([zero; 3], [zero; 3], [zero; 3])
    }
}
