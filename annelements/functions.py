
def std_to_pts(a, b, c, ref=[-1., 1.]):
    """
    Converts a 2D line from standard form (ax + by + c = 0) to two point form.
    Useful for plotting line equations in matplotlib.
    """
    if a == 0 and b == 0:
        raise Exception("std_to_pts: a and b can't both be zero.")

    (x1, y1), (x2, y2) = [(-c/a,p) if b==0 else (p, -((a*p + c)/b)) for p in ref]

    return ((x1, x2), (y1, y2))