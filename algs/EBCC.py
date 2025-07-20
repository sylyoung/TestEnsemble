# -*- coding: utf-8 -*-
# @Time    : 2025/7/18
# @Author  : Chenhao Liu
# @File    : EBCC.py
# Exploiting worker correlation for label aggregation in crowdsourcing
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma, gammaln
from scipy.stats import entropy, dirichlet


def ebcc_vb(preds,num_classes, num_groups=10, a_pi=0.1, alpha=1, a_v=4, b_v=1, max_iter=500, empirical_prior=False):
    num_items = preds.shape[1]
    num_workers = preds.shape[0]

    # Convert predictions to item-worker-label triplets
    first_col_repeated = np.repeat(np.arange(num_items), num_workers).reshape(-1)
    second_col = np.tile(np.arange(num_workers), num_items).reshape(-1)
    third_col = preds.T.flatten()
    tuples = np.vstack((first_col_repeated, second_col, third_col))
    tuples = tuples.T
    num_item, num_worker, num_class = tuples.max(axis=0) + 1

    y_is_one_lij = []
    y_is_one_lji = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        coo_ij = ssp.coo_matrix((np.ones(selected.sum()), tuples[selected, :2].T), shape=(num_items, num_workers),dtype=bool)
        y_is_one_lij.append(coo_ij.tocsr())
        y_is_one_lji.append(coo_ij.T.tocsr())

    # initialize confusion matrix
    beta_kl = np.eye(num_classes) * (a_v - b_v) + b_v

    # initialize z_ik, zg_ikm
    z_ik = np.zeros((num_items, num_classes))
    for l in range(num_classes):
        z_ik[:, [l]] += y_is_one_lij[l].sum(axis=-1)
    z_ik /= z_ik.sum(axis=-1, keepdims=True)

    if empirical_prior:
        alpha = z_ik.sum(axis=0)

    zg_ikm = np.random.dirichlet(np.ones(num_groups), z_ik.shape) * z_ik[:, :, None]
    for it in range(max_iter):
        # Update latent group distributions
        eta_km = a_pi / num_groups + zg_ikm.sum(axis=0)
        nu_k = alpha + z_ik.sum(axis=0)

        # Update confusion matrices
        mu_jkml = np.zeros((num_workers, num_classes, num_groups, num_classes)) + beta_kl[None, :, None, :]
        for l in range(num_classes):
            for k in range(num_classes):
                mu_jkml[:, k, :, l] += y_is_one_lji[l].dot(zg_ikm[:, k, :])

        # Compute expected log probabilities
        Eq_log_pi_km = digamma(eta_km) - digamma(eta_km.sum(axis=-1, keepdims=True))
        Eq_log_tau_k = digamma(nu_k) - digamma(nu_k.sum())
        Eq_log_v_jkml = digamma(mu_jkml) - digamma(mu_jkml.sum(axis=-1, keepdims=True))

        # Update worker group memberships
        zg_ikm[:] = Eq_log_pi_km[None, :, :] + Eq_log_tau_k[None, :, None]
        for l in range(num_classes):
            for k in range(num_classes):
                zg_ikm[:, k, :] += y_is_one_lij[l].dot(Eq_log_v_jkml[:, k, :, l])

        zg_ikm = np.exp(zg_ikm)
        zg_ikm /= zg_ikm.reshape(num_items, -1).sum(axis=-1)[:, None, None]

        last_z_ik = z_ik
        z_ik = zg_ikm.sum(axis=-1)

        if np.allclose(last_z_ik, z_ik, atol=1e-3):
            break

    return z_ik