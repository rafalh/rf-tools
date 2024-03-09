pub(crate) type Vector3 = [f32; 3];
pub(crate) type Plane = [f32; 4];
pub(crate) type Matrix4 = [[f32; 4]; 4];
pub(crate) type Matrix3 = [[f32; 3]; 3];

pub(crate) fn get_vector_len(vec: &Vector3) -> f32 {
    vec.iter().map(|v| v * v).sum::<f32>().sqrt()
}

pub(crate) fn transform_vector(pt: &Vector3, t: &Matrix3) -> Vector3 {
    let (x, y, z) = (pt[0], pt[1], pt[2]);
    [
        x * t[0][0] + y * t[1][0] + z * t[2][0],
        x * t[0][1] + y * t[1][1] + z * t[2][1],
        x * t[0][2] + y * t[1][2] + z * t[2][2],
    ]
}

pub(crate) fn transform_point(pt: &Vector3, t: &Matrix3) -> Vector3 {
    // for transforms without translation it is the same as for vector
    transform_vector(pt, t)
}

pub(crate) fn transform_normal(n: &Vector3, t: &Matrix3) -> Vector3 {
    let tn = transform_vector(n, t);
    // normalize transformed vector
    let l = get_vector_len(&tn);
    [tn[0] / l, tn[1] / l, tn[2] / l]
}

pub(crate) fn compute_triangle_normal(p0: &Vector3, p1: &Vector3, p2: &Vector3) -> Vector3 {
    let t0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let t1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];

    let mut normal = [0_f32; 3];
    normal[0] = (t0[1] * t1[2]) - (t0[2] * t1[1]);
    normal[1] = (t0[2] * t1[0]) - (t0[0] * t1[2]);
    normal[2] = (t0[0] * t1[1]) - (t0[1] * t1[0]);

    let len = get_vector_len(&normal);
    normal[0] /= len;
    normal[1] /= len;
    normal[2] /= len;

    normal
}

pub(crate) fn compute_triangle_plane(p0: &Vector3, p1: &Vector3, p2: &Vector3) -> Plane {
    let [a, b, c] = compute_triangle_normal(p0, p1, p2);
    let d = -(a * p0[0] + b * p0[1] + c * p0[2]);
    [a, b, c, d]
}

pub(crate) fn generate_uv(pos: &Vector3, n: &Vector3) -> [f32; 2] {
    if n[0].abs() >= n[1].abs().max(n[2].abs()) {
        // X is greatest
        if n[0] >= 0.0 {
            // right side
            [pos[2], pos[1]]
        } else {
            // left side
            [pos[1], pos[2]]
        }
    } else if n[1].abs() >= n[0].abs().max(n[2].abs()) {
        // Y is greatest
        if n[1] >= 0.0 {
            // top side
            [pos[0], pos[2]]
        } else {
            // bottom side
            [pos[2], pos[0]]
        }
    } else {
        // Z is greatest
        if n[2] >= 0.0 {
            // front side
            [pos[1], pos[0]]
        } else {
            // back side
            [pos[0], pos[1]]
        }
    }
}
