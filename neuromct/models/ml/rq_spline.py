"""
Monotonic rational-quadratic spline transforms (Durkan et al., "Neural Spline
Flows", NeurIPS 2019), specialized to the 1D conditional setting used by NFDE.

The transform is an element-wise monotone bijection of the real line: a
rational-quadratic spline with K bins on the interval [left, right] and the
identity outside of it (boundary derivatives are fixed to 1 so the map is
C^1 at the interval edges). Both directions are analytic -- in particular the
inverse requires no iterative root-finding, unlike the planar flow.

Shapes: x is [batch, n_values]; the raw (unconstrained) spline parameters
produced by a conditioner network are w_raw, h_raw of shape [batch, K] and
d_raw of shape [batch, K-1]; each batch row shares one spline across its
n_values points, matching NFDE's conditioning structure.
"""
import torch
import torch.nn.functional as F

DEFAULT_MIN_BIN = 1e-3
DEFAULT_MIN_DERIV = 1e-3


def _normalize_spline_params(w_raw, h_raw, d_raw, left, right,
                             min_bin=DEFAULT_MIN_BIN,
                             min_deriv=DEFAULT_MIN_DERIV):
    """Map unconstrained conditioner outputs to valid knot positions and
    derivatives. Returns cumwidths/cumheights [B, K+1] and derivs [B, K+1]
    (boundary derivatives fixed to 1 for identity tails)."""
    n_bins = w_raw.shape[-1]
    widths = F.softmax(w_raw, dim=-1) * (1 - n_bins * min_bin) + min_bin
    heights = F.softmax(h_raw, dim=-1) * (1 - n_bins * min_bin) + min_bin

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = left + (right - left) * cumwidths
    cumwidths[..., -1] = right                       # exact right edge

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = left + (right - left) * cumheights
    cumheights[..., -1] = right

    derivs = min_deriv + F.softplus(d_raw)           # [B, K-1] interior knots
    ones = torch.ones_like(derivs[..., :1])
    derivs = torch.cat([ones, derivs, ones], dim=-1)  # [B, K+1]
    return cumwidths, cumheights, derivs


def rq_spline_transform(x, w_raw, h_raw, d_raw, left, right, inverse=False,
                        min_bin=DEFAULT_MIN_BIN, min_deriv=DEFAULT_MIN_DERIV):
    """Apply the spline (or its analytic inverse) element-wise.

    Parameters: x [B, n]; w_raw/h_raw [B, K]; d_raw [B, K-1].
    Returns (y [B, n], log_det [B, n]), where log_det is always the log of
    the FORWARD derivative dy/dx evaluated at the corresponding point (for
    the inverse direction, callers wanting d(x)/d(z) should negate it).
    Points outside [left, right] are mapped by the identity (log_det = 0).
    """
    cumwidths, cumheights, derivs = _normalize_spline_params(
        w_raw, h_raw, d_raw, left, right, min_bin, min_deriv)

    inside = (x > left) & (x < right)
    y = x.clone()
    log_det = torch.zeros_like(x)
    if not inside.any():
        return y, log_det

    # locate bins: by x-knots (forward) or y-knots (inverse)
    ref_knots = cumheights if inverse else cumwidths
    idx = (torch.searchsorted(ref_knots, x.contiguous(), right=True) - 1)
    idx = idx.clamp(0, cumwidths.shape[-1] - 2)

    xk = cumwidths.gather(-1, idx)
    wk = cumwidths.gather(-1, idx + 1) - xk
    yk = cumheights.gather(-1, idx)
    hk = cumheights.gather(-1, idx + 1) - yk
    dk = derivs.gather(-1, idx)
    dk1 = derivs.gather(-1, idx + 1)
    sk = hk / wk

    if not inverse:
        xi = ((x - xk) / wk).clamp(0.0, 1.0)
        xi1m = 1.0 - xi
        denom = sk + (dk1 + dk - 2 * sk) * xi * xi1m
        y_in = yk + hk * (sk * xi.pow(2) + dk * xi * xi1m) / denom
        ld_in = (torch.log(sk.pow(2) * (dk1 * xi.pow(2)
                                        + 2 * sk * xi * xi1m
                                        + dk * xi1m.pow(2)))
                 - 2 * torch.log(denom))
    else:
        dy = (x - yk).clamp(min=0.0)                 # here x plays the role of y
        a = hk * (sk - dk) + dy * (dk1 + dk - 2 * sk)
        b = hk * dk - dy * (dk1 + dk - 2 * sk)
        c = -sk * dy
        disc = (b.pow(2) - 4 * a * c).clamp(min=0.0)
        xi = (2 * c) / (-b - torch.sqrt(disc) - 1e-300)
        xi = xi.clamp(0.0, 1.0)
        xi1m = 1.0 - xi
        y_in = xk + xi * wk
        denom = sk + (dk1 + dk - 2 * sk) * xi * xi1m
        ld_in = (torch.log(sk.pow(2) * (dk1 * xi.pow(2)
                                        + 2 * sk * xi * xi1m
                                        + dk * xi1m.pow(2)))
                 - 2 * torch.log(denom))

    y = torch.where(inside, y_in, y)
    log_det = torch.where(inside, ld_in, log_det)
    return y, log_det
