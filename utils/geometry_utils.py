import numpy as np
import math

def calc_plane_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> float:
    """
    Compute the angle between the planes defined by points ABC and BCD.

    Args:
        A: np.ndarray, shape (3,), coordinates of point A.
        B: np.ndarray, shape (3,), coordinates of point B.
        C: np.ndarray, shape (3,), coordinates of point C.
        D: np.ndarray, shape (3,), coordinates of point D.

    Returns:
        float: Angle in degrees between the two planes, within [0, 180].
    """

    def vector_subtract(P, Q):
        # Subtract point Q from point P
        return (P[0] - Q[0], P[1] - Q[1], P[2] - Q[2])

    def cross_product(u, v):
        # Compute the cross product of vectors u and v
        return (u[1]*v[2] - u[2]*v[1],
                u[2]*v[0] - u[0]*v[2],
                u[0]*v[1] - u[1]*v[0])

    def dot_product(u, v):
        # Compute the dot product of vectors u and v
        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

    def norm(v):
        # Compute the Euclidean norm of vector v
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    # Compute vectors in plane ABC
    AB = vector_subtract(B, A)
    AC = vector_subtract(C, A)
    # Normal to plane ABC
    n1 = cross_product(AB, AC)

    # Compute vectors in plane BCD
    BC = vector_subtract(C, B)
    BD = vector_subtract(D, B)
    # Normal to plane BCD
    n2 = cross_product(BC, BD)

    # Compute angle between n1 and n2 using dot product formula
    # Avoid division by zero error
    norm_n1 = norm(n1)
    norm_n2 = norm(n2)
    if norm_n1 == 0 or norm_n2 == 0:
        return 180
        raise ValueError("One of the planes is degenerate (points are collinear).")

    # Clamp dot_prod_ratio to [-1, 1] to avoid numeric issues
    dot_prod = dot_product(n1, n2)
    cos_angle = max(min(dot_prod / (norm_n1 * norm_n2), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)

    # Convert radian to degree
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_finger_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> float:
    """
    Compute the signed angle between CF and AC in plane E.
    Plane E is defined as the plane passing through line AC and the line through A
    with direction equal to the palm normal (from points A, B, C). D is projected onto plane E,
    meaning that CF is the projection of CD onto plane E.
    The angle is computed from AC (0° reference) to CF with a range of [-180, 180] degrees.

    Args:
        A: np.ndarray, shape (3,), point A on the palm.
        B: np.ndarray, shape (3,), point B on the palm.
        C: np.ndarray, shape (3,), point C on the palm.
        D: np.ndarray, shape (3,), fourth point for angle calculation.

    Returns:
        float: Signed angle in degrees between CF and AC.
    """
    # Compute palm plane normal from A, B, C
    n_palm = normalize(np.cross(B - A, C - A))

    # Calculate AC vector and its unit vector (reference direction)
    vec_AC = C - A
    ref = normalize(vec_AC)

    # Define plane E: it passes through A with two direction vectors: AC and n_palm.
    # Its out-of-plane normal can be computed as:
    n_E = normalize(np.cross(vec_AC, n_palm))

    # Project D onto plane E defined as: F = D - ((D-C) · n_E) * n_E, because C is on plane E
    v_CD = D - C
    F = D - np.dot(v_CD, n_E) * n_E

    # Compute the projection vector CF in plane E
    v_CF = F - C
    target = normalize(v_CF)

    # Compute signed angle between ref (AC) and target (CF)
    # The sign is determined by the direction of the cross product relative to n_E.
    cross_prod = np.cross(ref, target)
    sin_theta = np.dot(n_E, cross_prod)
    cos_theta = np.dot(ref, target)
    angle_rad = np.arctan2(sin_theta, cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


if __name__ == '__main__':
    # Example test for compute_finger_angle
    # Define sample points A, B, C (forming the palm) and D.
    A = np.array([0, 0, 0], dtype=np.float32)
    B = np.array([1, 0, 0], dtype=np.float32)
    C = np.array([0, 1, 0], dtype=np.float32)
    # Choose D such that its projection on plane E rotates relative to AC.
    D = np.array([0.5, 0.5, 1], dtype=np.float32)

    angle = compute_finger_angle(A, B, C, D)
    print(f"Signed angle between CF and AC: {angle:.2f} degrees")

    # Additional tests with different D positions:
    # D exactly along AC: expected angle 0
    D_along = C.copy()  # F would be C so CF becomes zero-length;
                       # in practice, a near zero vector should yield 0 angle.
    angle_along = compute_finger_angle(A, B, C, D_along)
    print(f"Signed angle (D along AC): {angle_along:.2f} degrees")

    # D rotated approximately 90 degrees in plane E relative to AC.
    D_rot = np.array([-0.5, 0.5, 1], dtype=np.float32)
    angle_rot = compute_finger_angle(A, B, C, D_rot)
    print(f"Signed angle (rotated): {angle_rot:.2f} degrees")


    A = np.array((0, 0, 0))
    B = np.array((1, 0, 0))
    C = np.array((0, 1, 0))
    D = np.array((0, 1, 1))

    angle = calc_plane_angle(A, B, C, D)
    print("The angle between planes ABC and BCD is:", angle, "degrees")

