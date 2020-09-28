# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, bad-continuation

'''
Module for optimization of functions with spherical parameters

Note:
    the content of this file is taken from:
        https://github.com/philippeller/spherical_opt/
    and pasted here to reduce the number of dependencies.
    All credit goes to P. Eller and contributors to spherical_opt.
'''

from __future__ import absolute_import, division, print_function

__author__ = 'P. Eller'
__license__ = '''Copyright 2019 Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.'''

import copy
import numpy as np

SPHER_T = np.dtype([
    ('zen', np.float32),
    ('az', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('sinzen', np.float32),
    ('coszen', np.float32),
    ('sinaz', np.float32),
    ('cosaz', np.float32),
])
"""type to store spherical coordinates and handy quantities"""


def fill_from_spher(s):
    """Fill in the remaining values in SPHER_T type giving the two angles `zen` and
    `az`.

    Parameters
    ----------
    s : SPHER_T or np.ndarray with dtype SPHER_T
    """
    s['sinzen'] = np.sin(s['zen'])
    s['coszen'] = np.cos(s['zen'])
    s['sinaz'] = np.sin(s['az'])
    s['cosaz'] = np.cos(s['az'])
    s['x'] = s['sinzen'] * s['cosaz']
    s['y'] = s['sinzen'] * s['sinaz']
    s['z'] = s['coszen']


def fill_from_cart(s_vector):
    """Fill in the remaining values in SPHER_T type giving the cart, coords. `x`, `y`
    and `z`.

    Parameters
    ----------
    s_vector : np.ndarray with dtype SPHER_T
    """
    radii = np.sqrt(s_vector['x']**2 + s_vector['y']**2 + s_vector['z']**2)

    pos_rad_inds = radii > 0
    s_vec_pos = s_vector[pos_rad_inds]
    pos_radii = radii[pos_rad_inds]

    # make sure they're length 1
    s_vec_pos['x'] /= pos_radii
    s_vec_pos['y'] /= pos_radii
    s_vec_pos['z'] /= pos_radii
    s_vec_pos['az'] = np.arctan2(s_vec_pos['y'], s_vec_pos['x']) % (2 * np.pi)
    s_vec_pos['coszen'] = s_vec_pos['z']
    s_vec_pos['zen'] = np.arccos(s_vec_pos['coszen'])
    s_vec_pos['sinzen'] = np.sin(s_vec_pos['zen'])
    s_vec_pos['sinaz'] = np.sin(s_vec_pos['az'])
    s_vec_pos['cosaz'] = np.cos(s_vec_pos['az'])

    s_vec_zero = s_vector[~pos_rad_inds]
    # print('zero length')
    s_vec_zero['z'] = 1.
    s_vec_zero['az'] = 0.
    s_vec_zero['zen'] = 0.
    s_vec_zero['coszen'] = 1.
    s_vec_zero['sinzen'] = 0.
    s_vec_zero['cosaz'] = 1.
    s_vec_zero['sinaz'] = 0.

    s_vector[pos_rad_inds] = s_vec_pos
    s_vector[~pos_rad_inds] = s_vec_zero


def reflect(old, centroid, new):
    """Reflect the old point around the centroid into the new point on the sphere.

    Parameters
    ----------
    old : SPHER_T or np.ndarray with dtype SPHER_T
    centroid : SPHER_T or np.ndarray with dtype SPHER_T
    new : SPHER_T or np.ndarray with dtype SPHER_T
    """
    x = old['x']
    y = old['y']
    z = old['z']

    ca = centroid['cosaz']
    sa = centroid['sinaz']
    cz = centroid['coszen']
    sz = centroid['sinzen']

    new['x'] = (
        2*ca*cz*sz*z
        + x*(ca*(-ca*cz**2 + ca*sz**2) - sa**2)
        + y*(ca*sa + sa*(-ca*cz**2 + ca*sz**2))
    )
    new['y'] = (
        2*cz*sa*sz*z
        + x*(ca*sa + ca*(-cz**2*sa + sa*sz**2))
        + y*(-ca**2 + sa*(-cz**2*sa + sa*sz**2))
    )
    new['z'] = 2*ca*cz*sz*x + 2*cz*sa*sz*y + z*(cz**2 - sz**2)

    fill_from_cart(new)


def centroid(cart_coords, sph_coord, axis=0):
    """Compute centroid of two or more points

    Parameters
    ----------
    cart_coords : np.ndarray
    sph_coord : np.ndarray of type SPHER_T
    axis : int
        axis along which to conduct the average
    """
    # determine output shape and prepare output array
    centroid_sph_shape = list(sph_coord.shape)
    del centroid_sph_shape[axis]
    centroid_sph = np.zeros(centroid_sph_shape, dtype=SPHER_T)

    for dim in ['x', 'y', 'z']:
        centroid_sph[dim] = np.sum(sph_coord[dim], axis=axis) / sph_coord.shape[axis]

    fill_from_cart(centroid_sph)

    centroid_cart = np.sum(cart_coords, axis=axis) / cart_coords.shape[axis]

    return centroid_cart, centroid_sph


def angular_dist(p1, p2): # theta1, theta2, phi1, phi2):
    """
    calculate the angular distance between two directions in spherical coords
    """
    return np.arccos(p1['coszen'] * p2['coszen'] + p1['sinzen'] * p2['sinzen'] * np.cos(p1['az'] - p2['az']))


def find_replacements(old_list, new_list, sorted_old_inds=None, sorted_new_inds=None):
    """find elements of new_list that are smaller than elements of old_list

    Helps replace elements of old list with elements of new list that are smaller.

    If there are N replacements, the N lowest values from new_list will replace the
    N highest values from old_list, and each replaced old value will be paired with
    a replacement new value that is smaller.

    The number of replacements will be as large as possible while respecting the above conditions.

    Within-list element ordering is determined via np.argsort if sorted_<old/new>_inds is None,
    otherwise it is determined by the provided sorted_<old/new>_inds parameters.

    the length of new_list must be less than or equal to the length of old_list

    Returns
    -------
    list of tuples
        list of tuples of the form (old_ind, new_ind), meaning that the item at index old_ind
        in old_list should be replaced with the item at index new_ind in new_list
    """

    if sorted_old_inds is None:
        sorted_old_inds = np.argsort(old_list)
    if sorted_new_inds is None:
        sorted_new_inds = np.argsort(new_list)

    max_N = len(sorted_new_inds)

    # Determine the maximum number of possible replacements
    for N_to_replace in range(max_N, 0, -1):
        # N_to_replace worst old values in increasing order
        old_inds = sorted_old_inds[-N_to_replace:]
        # N_to_replace best new values
        new_inds = sorted_new_inds[:N_to_replace]

        # we can replace N_to_replace values if and only if
        # new[new_inds[i]] < old[old_inds[i]] for all i
        less_thans = new_list[new_inds] < old_list[old_inds]
        if np.count_nonzero(less_thans) == N_to_replace:
            return list(zip(old_inds, new_inds))

    return []


def spherical_opt(
    func,
    method,
    initial_points,
    spherical_indices=tuple(),
    batch_size=1,
    max_iter=10000,
    max_calls=None,
    max_noimprovement=1000,
    fstdthresh=1e-1,
    cstdthresh=None,
    sstdthresh=None,
    meta=False,
    verbose=False,
    rand=None,
):
    """spherical minimization

    Parameters:
    -----------
    func : callable
        objective function
        if batch_size == 1, func should be a scalar function
        if batch_size >  1, func should be a vector function
    method : string
        choices of 'Nelder-Mead' and 'CRS2'
    inital_points : array
        providing the initial points for the algorithm, shape (N_points, N_dim)
    spherical_indices : iterable of tuples
        indices of spherical coordinates in pairs of (azimuth, zenith) e.g.
        `[[0,1], [7,8]]` would identify indices 0 as azimuth and 1 as zenith as
        spherical coordinates and 7 and 8 another pair of independent spherical
        coordinates
    batch_size : int, optional
        the number of new points proposed at each algorithm iteration
        batch_size > 1 is only supported for the CRS2 method
    max_iter : int
        maximum number of iterations
    max_calls : int
        maximum number of function calls
    max_noimprovement : int
        break condition, maximum iterations without improvement
    fstdthresh : float
        break condition, if std(f(p_i)) for all current points p_i drops below
        fstdthresh, minimization terminates
    cstdthresh : array
        break condition, if std(p_i) for all non-spherical coordinates current
        points p_i drops below cstdthresh, minimization terminates, for
        negative values, coordinate will be ignored
    sstdthresh : array
        break condition, if std(p_i) for all spherical coordinates current
        points p_i drops below sstdthresh, minimization terminates, for
        negative values, coordinate will be ignored
    verbose : bool
    rand : numpy random state or generator (optional)

    Notes
    -----
    CRS2 [1] is a variant of controlled random search (CRS, a global
    optimizer) with faster convergence than CRS.

    References
    ----------
    .. [1] P. Kaelo, M.M. Ali, "Some variants of the controlled random
       search algorithm for global optimization," J. Optim. Theory Appl.,
       130 (2) (2006), pp. 253-264.
    """
    if not method in ['Nelder-Mead', 'CRS2']:
        raise ValueError('Unknown method %s, choices are Nelder-Mead or CRS2'%method)

    if batch_size > 1:
        if method != 'CRS2':
            raise ValueError(
                'batch_size > 1 is only supported for the CRS2 method!'
                f' You selected the {method} method.')
        vec_func = func
    else:
        def vec_func(x):
            return np.array([func(x_i) for x_i in x])

    if rand is None:
        rand = np.random.default_rng()

    #REPORT_AFTER = 100

    n_points, n_dim = initial_points.shape
    n_spher = len(spherical_indices)
    n_cart = n_dim - 2 * n_spher

    sstd = np.full(n_spher, fill_value=-1)
    cstd = np.full(n_cart, fill_value=-1)

    if meta:
        meta_dict = {}

    if method == 'Nelder-Mead':
        assert n_points == n_dim + 1, 'Nelder-Mead will need n+1 points for an n-dimensional function'

    if method == 'CRS2':
        assert n_points > n_dim, 'CRS will need more points than dimensions'
        assert (n_points - 1) / n_dim >= batch_size, f'(n_points - 1) / n_dim must be >= batch size ({batch_size})!'

        if n_points < 10 * n_dim:
            print('WARNING: number of points is very low')

        if meta:
            meta_dict['num_simplex_successes'] = 0
            meta_dict['num_mutation_successes'] = 0
            meta_dict['num_failures'] = 0

    if cstdthresh is not None:
        assert len(cstdthresh) == n_cart, 'Std-dev stopping values for Cartesian coordinates must have length equal to number of Cartesian coordinates'
        cstdthresh = np.array(cstdthresh)
        cstdthresh_gtz_mask = cstdthresh > 0
        if np.count_nonzero(cstdthresh_gtz_mask) == 0:
            cstdthresh = None
        elif meta:
            meta_dict['cstdthresh_met_at_iter'] = np.full(np.count_nonzero(cstdthresh_gtz_mask), -1)

    if sstdthresh is not None:
        assert len(sstdthresh) == n_spher, 'Std-dev stopping values for spherical coordinates must have length equal to number of spherical coordinate pairs'
        sstdthresh = np.array(sstdthresh)
        sstdthresh_gtz_mask = sstdthresh > 0
        if np.count_nonzero(sstdthresh_gtz_mask) == 0:
            sstdthresh = None
        elif meta:
            meta_dict['sstdthresh_met_at_iter'] = np.full(np.count_nonzero(sstdthresh_gtz_mask), -1)

    all_spherical_indices = [idx for sp in spherical_indices for idx in sp]
    all_azimuth_indices = [sp[0] for sp in spherical_indices]
    all_zenith_indices = [sp[1] for sp in spherical_indices]
    all_cartesian_indices = list(set(range(n_dim)) ^ set(all_spherical_indices))

    # first thing, pack the points into separate cartesian and spherical coordinates
    fvals = vec_func(initial_points)

    s_cart = initial_points[:, all_cartesian_indices]
    #print(s_cart)
    s_spher = np.zeros(shape=(n_points, n_spher), dtype=SPHER_T)
    s_spher['az'] = initial_points[:, all_azimuth_indices]
    s_spher['zen'] = initial_points[:, all_zenith_indices]
    fill_from_spher(s_spher)

    # the array containing points in the original form
    x = copy.copy(initial_points)

    def create_x(x_cart, x_spher):
        '''Patch Cartesian and spherical coordinates back together into one array for function calls'''
        if len(x_cart.shape) == 1:
            # scalar case
            x = np.empty(shape=(n_dim))
            x[all_cartesian_indices] = x_cart
            x[all_azimuth_indices] = x_spher['az']
            x[all_zenith_indices] = x_spher['zen']
        else:
            # batch/vector case
            x = np.empty(shape=(x_cart.shape[0], n_dim))
            x[:, all_cartesian_indices] = x_cart
            x[:, all_azimuth_indices] = x_spher['az']
            x[:, all_zenith_indices] = x_spher['zen']

        return x

    best_fval = np.min(fvals)
    best_idx = 0
    no_improvement_counter = -1
    n_calls = n_points
    stopping_flag = -1

    # minimizer loop
    for iter_num in range(max_iter+1):
        #print(iter_num)

        if max_calls and n_calls >= max_calls:
            stopping_flag = 0
            break

        # break condition 2
        if max_noimprovement and no_improvement_counter > max_noimprovement:
            stopping_flag = 2
            break

        # break condition 1
        if np.std(fvals) < fstdthresh:
            stopping_flag = 1
            break

        # break condition 3
        if cstdthresh is not None or sstdthresh is not None:
            if cstdthresh is not None:
                cstd[:] = np.std(s_cart, axis=0)
                converged = cstd[cstdthresh_gtz_mask] < cstdthresh[cstdthresh_gtz_mask]
                if meta:
                    mask = np.logical_and(meta_dict['cstdthresh_met_at_iter'] < 0, converged)
                    meta_dict['cstdthresh_met_at_iter'][mask] = iter_num
                converged = np.all(converged)
            else:
                converged = True

            # TODO: stddev in spherical coords.
            if sstdthresh is not None:
                thresh_idx = 0
                for sph_pair_idx, stdthresh in enumerate(sstdthresh):
                    if stdthresh > 0:
                        _, cent = centroid(np.empty([0, 0]), s_spher)
                        deltas = angular_dist(s_spher, cent)
                        std = np.sqrt(np.sum(np.square(deltas)) / (n_points - 1))
                        sstd[sph_pair_idx] = std
                        converged = converged and std < stdthresh
                        if meta:
                            if meta_dict['sstdthresh_met_at_iter'][thresh_idx] < 0 and std < stdthresh:
                                meta_dict['sstdthresh_met_at_iter'][thresh_idx] = iter_num
                        thresh_idx += 1
                    else:
                        sstd[i] = -1

            if converged:
                stopping_flag = 3
                break

        sorted_idx = np.argsort(fvals)
        worst_idx = sorted_idx[-1]
        best_idx = sorted_idx[0]

        new_best_fval = fvals[best_idx]
        if new_best_fval < best_fval:
            best_fval = new_best_fval
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if method == 'CRS2':

            # shuffle into batch_size groups of n_dim points,
            # not including the best point
            batch_indices = rand.choice(n_points - 1, (batch_size, n_dim), replace=False)
            batch_indices[batch_indices >= best_idx] += 1

            # --- STEP 1: Reflection ---

            # centroid of choices except last, but including best
            centroid_indices = copy.copy(batch_indices)
            centroid_indices[:, -1] = best_idx

            cart_pts = s_cart[centroid_indices.flat].reshape((batch_size, n_dim, n_cart))
            spher_pts = s_spher[centroid_indices.flat].reshape((batch_size, n_dim, n_spher))
            # shape is (batch_size, n_dim, n_cart/n_sph). We want to average along axis 1
            centroid_cart, centroid_spher = centroid(cart_pts, spher_pts, axis=1)

            # create reflected points
            reflected_p_carts = 2 * centroid_cart - s_cart[batch_indices[:, -1]]
            reflected_p_sphers = np.zeros((batch_size, n_spher), dtype=SPHER_T)
            reflect(s_spher[batch_indices[:, -1]], centroid_spher, reflected_p_sphers)

            pts_to_eval = create_x(reflected_p_carts, reflected_p_sphers)
            new_fvals = vec_func(pts_to_eval)
            n_calls += batch_size

            # replace old points with new points that have lower values of func
            sorted_new = np.argsort(new_fvals)
            replacements = find_replacements(fvals, new_fvals, sorted_idx, sorted_new)
            for replace_ind, new_ind in replacements:
                s_cart[replace_ind] = reflected_p_carts[new_ind]
                s_spher[replace_ind] = reflected_p_sphers[new_ind]
                x[replace_ind] = pts_to_eval[new_ind]
                fvals[replace_ind] = new_fvals[new_ind]

            n_simplex_replaces = len(replacements)

            if meta:
                meta_dict['num_simplex_successes'] += n_simplex_replaces

            if n_simplex_replaces == batch_size:
                # no points left for mutation;
                # continue to next iteration
                continue

            # --- STEP 2: Mutation ---

            inds_to_mutate = sorted_new[n_simplex_replaces:]
            # do not replace points created in this iteration
            inds_to_replace = sorted_idx[:len(sorted_idx) - n_simplex_replaces]
            p_cart_to_mutate = reflected_p_carts[inds_to_mutate]
            p_spher_to_mutate = reflected_p_sphers[inds_to_mutate]
            n_to_mutate = len(p_cart_to_mutate)

            # update best_idx for mutation
            best_idx = np.argmin(fvals)

            w = rand.uniform(0, 1, (n_to_mutate, n_cart))
            mutated_p_carts = (1 + w) * s_cart[best_idx] - w * p_cart_to_mutate

            # first reflect at best point
            help_p_spher = np.zeros((n_to_mutate, n_spher), dtype=SPHER_T)
            reflect(p_spher_to_mutate, s_spher[best_idx], help_p_spher)

            # now do a combination of best and reflected point with random weights
            mutated_p_sphers = np.zeros_like(help_p_spher)
            w = rand.uniform(0, 1, (3, n_to_mutate, n_spher))
            for i, dim in enumerate(['x', 'y', 'z']):
                w_i = w[i]
                mutated_p_sphers[dim] = (1 - w_i) * s_spher[best_idx][dim] + w_i * help_p_spher[dim]
            fill_from_cart(mutated_p_sphers)

            pts_to_eval = create_x(mutated_p_carts, mutated_p_sphers)
            new_fvals = vec_func(pts_to_eval)
            n_calls += n_to_mutate

            # replace old points with new points that have lower values of func
            replacements = find_replacements(fvals, new_fvals, inds_to_replace)
            for replace_ind, new_ind in replacements:
                s_cart[replace_ind] = mutated_p_carts[new_ind]
                s_spher[replace_ind] = mutated_p_sphers[new_ind]
                x[replace_ind] = pts_to_eval[new_ind]
                fvals[replace_ind] = new_fvals[new_ind]

            n_mutation_replaces = len(replacements)

            if meta:
                meta_dict['num_mutation_successes'] += n_mutation_replaces

                if n_simplex_replaces == n_mutation_replaces == 0:
                    meta_dict['num_failures'] += 1

        elif method == 'Nelder-Mead':

            # --- STEP 1: Reflection ---
            if verbose: print('reflect')
            # centroid of choice except N+1, but including best
            centroid_indices = sorted_idx[:-1]
            centroid_cart, centroid_spher = centroid(s_cart[centroid_indices], s_spher[centroid_indices])

            # reflect point
            reflected_p_cart = 2 * centroid_cart - s_cart[worst_idx]
            reflected_p_spher = np.zeros(n_spher, dtype=SPHER_T)
            reflect(s_spher[worst_idx], centroid_spher, reflected_p_spher)
            reflected_p = create_x(reflected_p_cart, reflected_p_spher)
            reflected_fval = func(reflected_p)
            n_calls += 1

            if reflected_fval < fvals[sorted_idx[-2]] and reflected_fval >= fvals[best_idx]:
                # found better point
                s_cart[worst_idx] = reflected_p_cart
                s_spher[worst_idx] = reflected_p_spher
                x[worst_idx] = reflected_p
                fvals[worst_idx] = reflected_fval
                continue

            # --- STEP 2: Expand ---

            if reflected_fval < fvals[best_idx]:
                if verbose: print('expand')

                # essentially reflect again
                expanded_p_spher = np.zeros(n_spher, dtype=SPHER_T)
                reflect(centroid_spher, reflected_p_spher, expanded_p_spher)
                expanded_p_cart = 2. * reflected_p_cart - centroid_cart
                expanded_p = create_x(expanded_p_cart, expanded_p_spher)
                expanded_fval = func(expanded_p)
                n_calls += 1

                if expanded_fval < reflected_fval:
                    s_cart[worst_idx] = expanded_p_cart
                    s_spher[worst_idx] = expanded_p_spher
                    x[worst_idx] = expanded_p
                    fvals[worst_idx] = expanded_fval
                else:
                    s_cart[worst_idx] = reflected_p_cart
                    s_spher[worst_idx] = reflected_p_spher
                    x[worst_idx] = reflected_p
                    fvals[worst_idx] = reflected_fval
                continue

            # --- STEP 3: Contract ---

            if reflected_fval < fvals[worst_idx]:
                if verbose: print('contract (outside)')
                contracted_p_cart, contracted_p_spher = centroid(np.vstack([centroid_cart, reflected_p_cart]), np.vstack([centroid_spher, reflected_p_spher]))
                contracted_p = create_x(contracted_p_cart, contracted_p_spher)
                contracted_fval = func(contracted_p)
                n_calls += 1
                if contracted_fval < reflected_fval:
                    s_cart[worst_idx] = contracted_p_cart
                    s_spher[worst_idx] = contracted_p_spher
                    x[worst_idx] = contracted_p
                    fvals[worst_idx] = contracted_fval
                    continue
            else:
                if verbose: print('contract (inside)')
                contracted_p_cart, contracted_p_spher = centroid(np.vstack([centroid_cart, s_cart[worst_idx]]), np.vstack([centroid_spher, s_spher[worst_idx]]))
                contracted_p = create_x(contracted_p_cart, contracted_p_spher)
                contracted_fval = func(contracted_p)
                n_calls += 1
                if contracted_fval < fvals[worst_idx]:
                    s_cart[worst_idx] = contracted_p_cart
                    s_spher[worst_idx] = contracted_p_spher
                    x[worst_idx] = contracted_p
                    fvals[worst_idx] = contracted_fval
                    continue

            # --- STEP 4: Shrink ---
            if verbose: print('shrink')

            for idx in range(n_points):
                if not idx == best_idx:
                    s_cart[idx], s_spher[idx] = centroid(s_cart[[best_idx, idx]], s_spher[[best_idx, idx]])
                    x[idx] = create_x(s_cart[idx], s_spher[idx])
                    fvals[idx] = func(x[idx])
                    n_calls += 1


    opt_meta = {}
    opt_meta['stopping_flag'] = stopping_flag
    opt_meta['n_calls'] = n_calls
    opt_meta['nit'] = iter_num
    opt_meta['method'] = method
    opt_meta['fun'] = fvals[best_idx]
    opt_meta['x'] = x[best_idx]
    opt_meta['final_simplex'] = [x, fvals]
    opt_meta['success'] = stopping_flag > 0

    if meta:
        # Must re-compute all things here since some might not be computed (can
        # make this more performant by only computing things NOT computed for
        # stop condition to be met)

        # Standard deviation of cartesian coordinates
        cstd[:] = np.std(s_cart, axis=0)

        # Standard deviation of spherical coordinate pairs
        for i in range(n_spher):
            _, cent = centroid(np.empty([0, 0]), s_spher)
            deltas = angular_dist(s_spher, cent)
            sstd[i] = np.sqrt(np.sum(np.square(deltas)) / (n_points - 1))

        # Standard deviation of func return value (e.g., metric)
        fstd = np.std(fvals)

        meta_dict['no_improvement_counter'] = no_improvement_counter
        meta_dict['fstd'] = fstd
        meta_dict['cstd'] = cstd
        meta_dict['sstd'] = sstd

        opt_meta['meta'] = meta_dict

    return opt_meta

