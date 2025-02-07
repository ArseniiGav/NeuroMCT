"""
This script is based on the scipy.stats._cdf_distance.py implementation
for the statistical distances. The original code can be found at:
https://github.com/scipy/scipy/blob/v1.14.1/scipy/stats/_stats_py.py#L10210

The code was adapted to PyTorch and unified to work
with differrent p-norms in a signle nn.Module-based class.
"""
import torch
import torch.nn as nn


class LpNormDistance(nn.Module):
    def __init__(self, p=2, reduction='mean'):
        r"""
        The Pytorch module to compute the Lp-norm distance 
        between two one-dimensional probability distributions.

        Parameters
            - p : int, float, or torch.inf, optional (default=2)
                The order of the norm:
                - If p = 1, computes the Wasserstein distance.
                - If p = 2, computes the Cramér-von Mises distance.
                - If p = torch.inf, computes the Kolmogorov-Smirnov distance.
                - if p is a positive float computer the general Lp distance.
        Methods
            - forward(u_values, v_values, u_weights=None, v_weights=None)
                Computes Lp-norm distance between two one-dimensional probability distributions u and v,
                whose respective CDFs are U and V. Where the Lp-norm distance that is defined as follows:

                .. math::

                    l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

                p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
                gives the Cramér-von Mises distance.

                Parameters
                ----------
                u_values, v_values: array_like
                    Values observed in the (empirical) distribution.
                u_weights, v_weights: array_like, optional
                    Weight for each value. If unspecified, each value is assigned the same weight.
                    u_weights (resp. v_weights) must have the same length as u_values (resp. v_values). 

                Returns
                -------
                distance : float
                    The computed distance between the distributions.
        """
        super().__init__()
        self.reduction = reduction
        self.p = torch.tensor(p)
        if torch.isfinite(self.p) and self.p <= 0:
            raise ValueError("p must be positive: p > 0.")
        elif not torch.isfinite(self.p) and torch.sign(self.p) == -1:
            raise ValueError("p cannot be neg. infinity, p must be positive: p > 0.")

    def _compute_cdf(self, values, weights, all_values, batch_size):
        #torch.cumsum does not have a deterministic implementation in torch 2.4.0
        original_deterministic_setting = torch.are_deterministic_algorithms_enabled()
        if original_deterministic_setting:
            torch.use_deterministic_algorithms(True, warn_only=True)
        indices = torch.searchsorted(values, all_values[:, :-1], right=True)
        if weights is None:
            cdf = indices / values.size(1)
        else:
            # Compute weighted CDF
            sorted_cumweights = torch.cat(
                [torch.zeros(batch_size, 1, device=values.device), torch.cumsum(weights, dim=1)],
                dim=1,
            )
            cdf = sorted_cumweights.gather(1, indices) / sorted_cumweights[:, -1:]
        return cdf

    def _get_cdf_diffs(self, u_values, v_values, u_weights=None, v_weights=None):
        batch_size = u_values.size(0)
        
        # Sort values
        u_sorted, _ = torch.sort(u_values, dim=1)
        v_sorted, _ = torch.sort(v_values, dim=1)
        
        # Combine all values and sort
        all_values = torch.cat([u_sorted, v_sorted], dim=1)
        all_values, _ = torch.sort(all_values, dim=1)
        
        # Compute differences between consecutive values
        x_deltas = torch.diff(all_values)
        
        # Calculate the CDFs for u and v
        u_cdf = self._compute_cdf(u_sorted, u_weights, all_values, batch_size)
        v_cdf = self._compute_cdf(v_sorted, v_weights, all_values, batch_size)
        
        # Compute the differences between CDFs
        cdfs_deltas = torch.abs(u_cdf - v_cdf)        
        return cdfs_deltas, x_deltas

    def forward(self, u_values, v_values, u_weights=None, v_weights=None):
        # Get the differences between CDFs and the the differences between the corresponding values
        cdfs_deltas, x_deltas = self._get_cdf_diffs(u_values, v_values, u_weights, v_weights)
        if torch.isfinite(self.p) and self.p > 0:
            distance = torch.pow(torch.sum((cdfs_deltas ** self.p) * x_deltas, dim=1), 1 / self.p)
        else:
            distance = torch.max(cdfs_deltas, dim=1).values
        
        if self.reduction == "mean":
            # normalize by the number of samples in the batch
            return distance.sum() / u_values.size(0)
        elif self.reduction == "sum":
             # sum the distances over the batch
            return distance.sum()
        elif self.reduction == "none":
            # no reduction
            return distance
        else:
            raise ValueError("reduction must be one of 'mean', 'sum', or 'none'.")
