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
    def __init__(self, p=2):
        """
        Initialize the LpNormDistance module.
        Args:
            p: int, float, or torch.inf
                - If p = 1, computes Wasserstein distance.
                - If p = 2, computes Cramér-von Mises distance.
                - If p = torch.inf, computes Kolmogorov-Smirnov distance.
        """
        super().__init__()
        self.p = p
    
    def _compute_cdf(self, values, weights, all_values, batch_size):
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

    def forward(self, x_values, y_values, x_weights=None, y_weights=None):
        r"""
        Compute, between two one-dimensional distributions :math:`u` and
        :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
        statistical distance that is defined as:

        .. math::

            l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

        p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
        gives the energy distance.

        Parameters
        ----------
        u_values, v_values : array_like
            Values observed in the (empirical) distribution.
        u_weights, v_weights : array_like, optional
            Weight for each value. If unspecified, each value is assigned the same
            weight.
            `u_weights` (resp. `v_weights`) must have the same length as
            `u_values` (resp. `v_values`). If the weight sum differs from 1, it
            must still be positive and finite so that the weights can be normalized
            to sum to 1.

        Returns
        -------
        distance : float
            The computed distance between the distributions.
        """
        # Get the differences between CDFs and the the differences between the corresponding values
        cdfs_deltas, x_deltas = self._get_cdf_diffs(x_values, y_values, x_weights, y_weights)
        if self.p == 1:
            # Wasserstein distance
            return torch.sum(cdfs_deltas * x_deltas, dim=1)
        elif self.p == 2:
            # Cramér-von Mises distance
            return torch.sqrt(torch.sum((cdfs_deltas ** 2) * x_deltas, dim=1))
        elif self.p == float("inf"):
            # Kolmogorov-Smirnov (KS) distance
            return torch.max(cdfs_deltas, dim=1).values
        else:
            # General Lp distance
            integral = torch.sum((cdfs_deltas ** self.p) * x_deltas, dim=1)
            return torch.pow(integral, 1 / self.p)
