#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Function
from .manifold import Manifold
import numpy as np
from torch.nn import Embedding
from torch.autograd import Function

class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        z = g / z
        return z, None

acosh = Acosh.apply

class LorentzManifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    def __init__(self, eps=1e-12, _eps=1e-5, norm_clip=1, max_norm=1e6,
            debug=False, K=None, **kwargs):
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug
        self.K = K
        if K is not None:
            self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))

    def allocate_lt(self, N, dim, sparse=False):
        return Embedding(N, dim + 1, sparse=sparse)

    def init_weights(self, w, init_range=1e-5):
        w.weight.data.uniform_(-init_range, init_range)
        self.normalize(w.weight.data)

    @staticmethod
    def ldot(u, v, keepdim=False):
        """Lorentzian Scalar Product"""
        uv = u * v
        uv = uv.narrow(-1, 0, 1).mul(-1)
        return torch.sum(uv, dim=-1, keepdim=keepdim)

    def to_poincare_ball(self, u):
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def distance(self, u, v):
        d = -LorentzDot.apply(u, v)
        d.data.clamp_(min=1)
        return acosh(d, self._eps)

    def norm(self, u):
        return torch.sqrt(torch.sum(torch.pow(self.to_poincare_ball(u), 2), dim=-1))

    def normalize(self, w):
        """Normalize vector such that it is located on the hyperboloid"""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = narrowed.view(-1, d).renorm(p=2, dim=0, maxnorm=self.max_norm)

        if self.K is not None:
            # Push embeddings outside of `inner_radius`
            w0 = w.narrow(-1, 0, 1).squeeze()
            wnrm = torch.sqrt(torch.sum(torch.pow(narrowed, 2), dim=-1)) / (1 + w0)
            scal = torch.ones_like(wnrm)
            ix = wnrm < (self.inner_radius + self._eps)
            scal[ix] = (self.inner_radius + self._eps) / wnrm[ix]
            narrowed = narrowed * scal.unsqueeze(-1)

        w0 = torch.sqrt(1 + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True))
        w = torch.cat([w0, narrowed], dim=-1)
        return w

    def normalize_tan(self, x_all, v_all):
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = torch.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + torch.sum(torch.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp = tmp.sqrt().clamp(min=self._eps)
        v_all = torch.cat([xv / tmp, v_all.narrow(1, 1, d)], dim=1)
        return v_all

    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        u = u.narrow(-1, 0, 1).mul(-1)
        u = u + self.ldot(x, u, keepdim=True).expand_as(x) * x
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for hyperboloid"""
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            p_val = self.normalize(p.index_select(0, ix))
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp(min=0).sqrt()
            t = torch.clamp(nd_p, max=self.norm_clip)
            nd_p = nd_p.clamp(min=self.eps)
            newp = (torch.cosh(t) * p_val) + (torch.sinh(t) * d_val) / nd_p
            if normalize:
                newp = self.normalize(newp)
            p = p.index_copy(0, ix, newp)
        else:
            if lr is not None:
                d_p = d_p.narrow(-1, 0, 1).mul(-1)
                d_p = d_p + (self.ldot(p, d_p, keepdim=True)).expand_as(p) * p
                d_p = d_p * (-lr)
            ldv = self.ldot(d_p, d_p, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp(min=0).sqrt()
            t = torch.clamp(nd_p, max=self.norm_clip)
            nd_p = nd_p.clamp(min=self.eps)
            newp = (torch.cosh(t) * p) + (torch.sinh(t) * d_p) / nd_p
            if normalize:
                newp = self.normalize(newp)
            p = newp

    def logm(self, x, y):
        """Logarithmic map on the Lorenz Manifold"""
        xy = torch.clamp(self.ldot(x, y).unsqueeze(-1), max=-1)
        v = acosh(-xy, self.eps) / (
            torch.clamp(torch.sqrt(xy * xy - 1), min=self._eps)
        ) * (y + xy * x)
        return self.normalize_tan(x, v)

    def ptransp(self, x, y, v, ix=None, out=None):
        """Parallel transport for hyperboloid"""
        if ix is not None:
            v_ = v
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        elif v.is_sparse:
            ix, v_ = v._indices().squeeze(), v._values()
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        else:
            raise NotImplementedError
        xy = self.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        if out is None:
            return vnew
        else:
            out.index_copy_(0, ix, vnew)

    def half_aperture(self, u):
        eps = self.eps
        d = u.size(-1) - 1
        sqnu = torch.sum(u.narrow(-1, 1, d) ** 2, dim=-1) / (1 + u.narrow(-1, 0, 1)
            .squeeze(-1)) ** 2
        sqnu.clamp_(min=0, max=1 - eps)
        return torch.asin((self.inner_radius * (1 - sqnu) / torch.sqrt(sqnu))
            .clamp(min=-1 + eps, max=1 - eps))

    def angle_at_u(self, u, v):
        uvldot = LorentzDot.apply(u, v)
        u0 = u.narrow(-1, 0, 1).squeeze(-1)
        num = torch.add(v.narrow(-1, 0, 1).squeeze(-1), torch.mul(u0, uvldot))
        tmp = torch.pow(uvldot, 2) - 1.
        den = torch.sqrt(torch.pow(u0, 2) - 1.) * torch.sqrt(tmp.clamp_(min=self.eps))
        frac = torch.div(num, den)
        # if self.debug and (frac != frac).any():
        #     import ipdb; ipdb.set_trace()
        frac.data.clamp_(min=-1 + self.eps, max=1 - self.eps)
        ksi = frac.acos()
        return ksi

    def norm(self, u):
        if isinstance(u, Embedding):
            u = u.weight
        d = u.size(-1) - 1
        sqnu = torch.sum(u.narrow(-1, 1, d) ** 2, dim=-1)
        sqnu = sqnu / (1 + u.narrow(-1, 0, 1).squeeze(-1)) ** 2
        return sqnu.sqrt()


class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        grad_u = v.clone()
        grad_u[..., 0] *= -1   # Minkowski sign flip on time coord

        grad_v = u.clone()
        grad_v[..., 0] *= -1

        grad_u = grad_u * g.unsqueeze(-1)
        grad_v = grad_v * g.unsqueeze(-1)

        grad_u = grad_u.sum_to_size(*u.shape)
        grad_v = grad_v.sum_to_size(*v.shape)

        return grad_u, grad_v
