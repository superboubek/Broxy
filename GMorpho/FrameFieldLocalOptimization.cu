/*
	Copyright (c) 2015-2017 Telecom ParisTech (France).
	Authors: Stephane Calderon and Tamy Boubekeur.
	All rights reserved.

	This file is part of Broxy, the reference implementation for
	the paper:
		Bounding Proxies for Shape Approximation.
		Stephane Calderon and Tamy Boubekeur.
		ACM Transactions on Graphics (Proc. SIGGRAPH 2017),
		vol. 36, no. 5, art. 57, 2017.

	You can redistribute it and/or modify it under the terms of the GNU
	General Public License as published by the Free Software Foundation,
	either version 3 of the License, or (at your option) any later version.

	Licensees holding a valid commercial license may use this file in
	accordance with the commercial license agreement provided with the software.

	This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
	WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "FrameField.h"

using namespace MorphoGraphics;

/******************************************************************************/
/* Local Optimization																													*/
/**************************************************************************** */

void FrameField::InitCUDAGCubeQuats () {
	for (int i = 0; i < 24; i++)
		gcube_quats_vec_[i]  = make_float4 ( 0,      0,      0,      1);

	// Beam symmetry group
	gcube_quats_vec_[0]  = make_float4 ( 0,      0,      0,      1);
	gcube_quats_vec_[1]  = make_float4 ( 1,      0,      0,      0);
	gcube_quats_vec_[2]  = make_float4 ( 0,      1,      0,      0);
	gcube_quats_vec_[3]  = make_float4 ( 0,      0,      1,      0);
	gcube_quats_vec_[4] = make_float4 ( dCPI4,  dCPI4,  0,      0);
	gcube_quats_vec_[5] = make_float4 ( 0,      0,      dCPI4,  dCPI4);
	gcube_quats_vec_[6] = make_float4 ( 0,      0,     -dCPI4,  dCPI4);
	gcube_quats_vec_[7] = make_float4 (-dCPI4,  dCPI4,  0,      0);

	// Cube symmetry group
	gcube_quats_vec_[8]  = make_float4 ( dCPI4,  0,      0,      dCPI4);
	gcube_quats_vec_[9]  = make_float4 (-dCPI4,  0,      0,      dCPI4);
	gcube_quats_vec_[10]  = make_float4 ( 0,      dCPI4,  dCPI4,  0);
	gcube_quats_vec_[11]  = make_float4 ( 0,     -dCPI4,  dCPI4,  0);

	gcube_quats_vec_[12]  = make_float4 ( 0.5,    0.5,    0.5,    0.5);
	gcube_quats_vec_[13]  = make_float4 (-0.5,   -0.5,    0.5,    0.5);
	gcube_quats_vec_[14] = make_float4 (-0.5,    0.5,    0.5,   -0.5);
	gcube_quats_vec_[15] = make_float4 ( 0.5,   -0.5,    0.5,   -0.5);

	gcube_quats_vec_[16] = make_float4 ( 0.5,    0.5,    0.5,   -0.5);
	gcube_quats_vec_[17] = make_float4 ( 0.5,   -0.5,    0.5,    0.5);
	gcube_quats_vec_[18] = make_float4 ( 0,     -dCPI4,  0,      dCPI4);
	gcube_quats_vec_[19] = make_float4 ( dCPI4,  0,      dCPI4,  0);
	gcube_quats_vec_[20] = make_float4 (-0.5,   -0.5,    0.5,   -0.5);
	gcube_quats_vec_[21] = make_float4 (-0.5,    0.5,    0.5,    0.5);
	gcube_quats_vec_[22] = make_float4 ( 0,      dCPI4,  0,  dCPI4);
	gcube_quats_vec_[23] = make_float4 (-dCPI4,  0,      dCPI4,  0);

	cudaMemcpyToSymbol (gcube_quats, &(gcube_quats_vec_[0].x),
	                    24 * 4 * sizeof (float));
	CheckCUDAError ();
}

void FrameField::InitBiHarmonicMasks () {
	int dim_mask = 5;
	Eigen::MatrixXf mat_5x5x5_l_host (dim_mask * dim_mask * dim_mask,
	                                  dim_mask * dim_mask * dim_mask);
	Eigen::MatrixXf mat_5x5x5_ltl_host (dim_mask * dim_mask * dim_mask,
	                                    dim_mask * dim_mask * dim_mask);

	std::vector<Vec3i> mask_ids;
	std::vector<float> mask_vals;

	mask_ids.clear ();
	mask_vals.clear ();
	mask_ids.push_back (Vec3i (0, 0, -1));
	mask_ids.push_back (Vec3i (0, -1, 0));
	mask_ids.push_back (Vec3i (-1, 0, 0));
	mask_ids.push_back (Vec3i (0, 0, 0));
	mask_ids.push_back (Vec3i (1, 0, 0));
	mask_ids.push_back (Vec3i (0, 1, 0));
	mask_ids.push_back (Vec3i (0, 0, 1));
	mask_vals.push_back (1.f);
	mask_vals.push_back (1.f);
	mask_vals.push_back (1.f);
	mask_vals.push_back (0.f);
	mask_vals.push_back (1.f);
	mask_vals.push_back (1.f);
	mask_vals.push_back (1.f);

	mat_5x5x5_l_host = Eigen::MatrixXf::Zero (dim_mask * dim_mask * dim_mask,
	                   dim_mask * dim_mask * dim_mask);
	mat_5x5x5_ltl_host = Eigen::MatrixXf::Zero (dim_mask * dim_mask * dim_mask,
	                     dim_mask * dim_mask * dim_mask);
	//	mat_5x5x5_ll_host = Eigen::MatrixXf::Zero (dim_mask*dim_mask*dim_mask,
	//																						 dim_mask*dim_mask*dim_mask);
	for (int k = 0; k < dim_mask; k++)
		for (int j = 0; j < dim_mask; j++)
			for (int i = 0; i < dim_mask; i++) {
				int cell_id = dim_mask * dim_mask * k + dim_mask * j + i;
				float sum_mask = 0.f;
				for (int l = 0; l < (int)mask_vals.size (); l++) {
					int im = i + mask_ids[l][0];
					int jm = j + mask_ids[l][1];
					int km = k + mask_ids[l][2];
					if ((0 <= im) && (im < dim_mask) &&
					        (0 <= jm) && (jm < dim_mask) &&
					        (0 <= km) && (km < dim_mask)) {
						sum_mask += mask_vals[l];
					}
				}

				for (int l = 0; l < (int)mask_vals.size (); l++) {
					int im = i + mask_ids[l][0];
					int jm = j + mask_ids[l][1];
					int km = k + mask_ids[l][2];
					int mask_cell_id = dim_mask * dim_mask * km + dim_mask * jm + im;
					if ((0 <= im) && (im < dim_mask) &&
					        (0 <= jm) && (jm < dim_mask) &&
					        (0 <= km) && (km < dim_mask)) {
						float mask_val;
						if (cell_id != mask_cell_id)
							mask_val = -(mask_vals[l] / sum_mask);
						else {
							mask_val = 1.f - (mask_vals[l] / sum_mask);
						}
						mat_5x5x5_l_host (cell_id, mask_cell_id) = mask_val;
					}
				}
			}


	// L^T*L BiLaplacian Template Matrix
	mat_5x5x5_ltl_host = mat_5x5x5_l_host.transpose () * mat_5x5x5_l_host;
	// L*L BiLaplacian Template Matrix
	//	mat_5x5x5_ll_host = mat_5x5x5_l_host*mat_5x5x5_l_host;

	// Compute Spectral Radius of the template Matrix
	// NB: important to ensure the convergence of the relaxation iterations
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver_5x5x5 (mat_5x5x5_ltl_host);

	if (eigensolver_5x5x5.info () != Eigen::Success) {
		std::cout << "impossible to compute eigen decomposition" << std::endl;
	} else {
		std::cout << "spectral radius : " << eigensolver_5x5x5.eigenvalues ().lpNorm<Eigen::Infinity> () << std::endl;
		std::cout << "max eigen value : " << eigensolver_5x5x5.eigenvalues ().maxCoeff () << std::endl;
		std::cout << "min eigen value : " << eigensolver_5x5x5.eigenvalues ().minCoeff () << std::endl;
	}

	bh_masks_spectral_radius_ = eigensolver_5x5x5.eigenvalues ().lpNorm<Eigen::Infinity> ();

	bh_masks_host_.resize (dim_mask * dim_mask * dim_mask);
	for (int k = 0; k < dim_mask; k++)
		for (int j = 0; j < dim_mask; j++)
			for (int i = 0; i < dim_mask; i++) {
				int cell_id = dim_mask * dim_mask * k + dim_mask * j + i;
				for (int k_n = -2; k_n <= 2; k_n++) {
					for (int j_n = -2; j_n <= 2; j_n++) {
						for (int i_n = -2; i_n <= 2; i_n++) {
							int im = i + i_n;
							int jm = j + j_n;
							int km = k + k_n;
							int mask_cell_id = dim_mask * dim_mask * km + dim_mask * jm + im;
							if ((0 <= im) && (im < dim_mask) &&
							        (0 <= jm) && (jm < dim_mask) &&
							        (0 <= km) && (km < dim_mask)) {
								bh_masks_host_[cell_id].push_back (mat_5x5x5_ltl_host (cell_id, mask_cell_id));
							} else {
								bh_masks_host_[cell_id].push_back (0.f);
							}
						}
					}
				}
			}

	std::vector<float> bh_masks_host_vec;
	bh_masks_host_vec.clear ();
	for (int i = 0; i < (bh_masks_host_.size ()); i++)
		for (int j = 0; j < (bh_masks_host_[i].size ()); j++) {
			bh_masks_host_vec.push_back (bh_masks_host_[i][j]);
		}

	cudaMemcpy (bh_masks_, &(bh_masks_host_vec[0]),
	            bh_masks_host_vec.size ()*sizeof (float),
	            cudaMemcpyHostToDevice);
	CheckCUDAError ();

	//	for (int k = 0; k < dim_mask; k++)
	//		for (int j = 0; j < dim_mask; j++)
	//			for (int i = 0; i < dim_mask; i++) {
	//				int cell_id = dim_mask*dim_mask*k + dim_mask*j + i;
	//				std::cout << "mask (" << i << ", " << j << ", " << k << ") : of size " << bh_masks[cell_id].size () << std::endl;
	//				for (int k_n = -2; k_n <=2; k_n++) {
	//					for (int j_n = -2; j_n <=2; j_n++) {
	//						for (int i_n = -2; i_n <=2; i_n++) {
	//							int im = i + i_n;
	//							int jm = j + j_n;
	//							int km = k + k_n;
	//							int mask_cell_id = dim_mask*dim_mask*km + dim_mask*jm + im;
	//							int mask_id_neigh = dim_mask*dim_mask*(2 + k_n) + dim_mask*(2 + j_n) + (2 + i_n);
	//							if ((0 <= im) && (im < dim_mask) &&
	//									(0 <= jm) && (jm < dim_mask) &&
	//									(0 <= km) && (km < dim_mask) || true) {
	//								std::cout << bh_masks[cell_id][mask_id_neigh] << " ";
	//							}
	//						}
	//						std::cout << std::endl;
	//					}
	//					std::cout << std::endl;
	//				}
	//				std::cout << std::endl;
	//			}
}

void FrameField::InitLocalOptimization (const Vec3f & bbox_min,
                                        const Vec3f & bbox_max,
                                        const Vec3i & res) {
	// Set Grid Info
	bbox_min_ = bbox_min;
	bbox_max_ = bbox_max;
	res_ = res;
	cell_size_ = max (fabs (bbox_max_[0] - bbox_min_[0]),
	                  max (fabs (bbox_max_[1] - bbox_min_[1]),
	                       fabs (bbox_max_[2] - bbox_min_[2])
	                      )
	                 ) / res_[0];

	slice_xy_id_ = 0;
	max_hc_number_ = 20;
	lambda_ = 10.f;

	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	int num_cells = dim_x * dim_y * dim_z;
	int num_hcs = hc_points_.size ();

	// CPU Data Allocation
	twist_field_ping_.resize (num_cells);
	position_field_ping_.resize (num_cells);
	quat_field_ping_.resize (num_cells);
	quat_field_pong_.resize (num_cells);
	w_hc_quats_host_.resize (num_cells);
	hc_quats_host_.resize (num_cells);

	float dx = cell_size_;
	float scale = dx;

	opengl_frame_field_p_.clear ();
	opengl_frame_field_n_.clear ();

	// Initialize CPU Data
	for (int i = 0; i < dim_x; i++)
		for (int j = 0; j < dim_y; j++)
			for (int k = 0; k < dim_z; k++) {
				Vec3f pos = scale * Vec3f (i, j, k)
				            + Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2])
				            + 0.5f * Vec3f (dx, dx, dx);
				position_field_ping_[i + dim_x * j + dim_x * dim_y * k] = pos;
				float4 rand_quat;
				//				rand_quat = make_float4 (((float)std::rand ())/RAND_MAX,
				//																 ((float)std::rand ())/RAND_MAX,
				//																 ((float)std::rand ())/RAND_MAX,
				//																 ((float)std::rand ())/RAND_MAX);
				//				rand_quat = normalize (rand_quat);

				//				if ((i == 8) && (j == 9) && (k == 4)) {
				//					//					rand_quat = make_float4 (0.1, 1.0, -1.2, 3.3);
				//					rand_quat = make_float4 (1.0, 0.1, 0.1, 0.1);
				//					//					rand_quat = make_float4 (1.0, 0, 0, 0);
				//					//					rand_quat = normalize (rand_quat);
				//					std::cout << "rand_quat : "
				//						<< rand_quat.x << " " << rand_quat.y << " " << rand_quat.z << " " << rand_quat.w << std::endl;
				//				} else
				//					rand_quat = make_float4 (1.f, 0.f, 0.f, 0.f);
				rand_quat = make_float4 (1.f, 0.f, 0.f, 0.f);
				quat_field_ping_[i + dim_x * j + dim_x * dim_y * k] = rand_quat;
				quat_field_pong_[i + dim_x * j + dim_x * dim_y * k] = rand_quat;
				hc_quats_host_[i + dim_x * j + dim_x * dim_y * k] = make_float4 (1.f, 0.f, 0.f, 0.f);
				w_hc_quats_host_[i + dim_x * j + dim_x * dim_y * k] = 0.f;
			}

	// Initialize OpenGL Data (Frame visualization)
	num_tris_in_cube_ = 12;
	opengl_frame_field_p_.reserve (3 * num_cells * num_tris_in_cube_);
	opengl_frame_field_n_.reserve (3 * num_cells * num_tris_in_cube_);
	ComputeOpenGLCube ();

	// GPU Data Allocation
	multi_grid_quats_.clear ();
	for (int i = num_cells; i > 0; i /= 8) {
		float4 * quats = NULL;
		cudaMalloc (&quats, i * sizeof (float4));
		CheckCUDAError ();
		multi_grid_quats_.push_back (quats);
	}

	cudaMalloc (&quats_pong_, num_cells * sizeof (float4));
	CheckCUDAError ();
	cudaMalloc (&quats_ping_, num_cells * sizeof (float4));
	CheckCUDAError ();
	cudaMalloc (&init_quats_, num_cells * sizeof (float4));
	CheckCUDAError ();
	cudaMalloc (&hc_quats_, num_cells * sizeof (float4));
	CheckCUDAError ();
	cudaMalloc (&w_hc_quats_, num_cells * sizeof (float));
	CheckCUDAError ();
	cudaMemset (hc_quats_, 0, num_cells * sizeof (float4));
	CheckCUDAError ();
	cudaMalloc (&bh_masks_, (5 * 5 * 5) * (5 * 5 * 5) * (5 * 5 * 5)*sizeof (float));
	CheckCUDAError ();
	cudaMemset (bh_masks_, 0, (5 * 5 * 5) * (5 * 5 * 5) * (5 * 5 * 5)*sizeof (float));
	CheckCUDAError ();

	// GPU Data Initialization
	InitCUDAGCubeQuats ();
	cudaMemset (w_hc_quats_, 0, num_cells * sizeof (float));
	CheckCUDAError ();
	cudaMemcpy (init_quats_, &(quat_field_ping_[0].x),
	            num_cells * sizeof (float4), cudaMemcpyHostToDevice);
	CheckCUDAError ();
	cudaMemcpy (quats_ping_, init_quats_,
	            num_cells * sizeof (float4), cudaMemcpyDeviceToDevice);
	CheckCUDAError ();

	// CPU & GPU Initialization of the BiHarmonic masks
	InitBiHarmonicMasks ();
}

__host__ __device__
float4 LocalSymmetryOptimization (const float4 group_quat[24],
                                  const float4 & q_a,
                                  const float4 & q_b) {
	float min_dist_q_b_a_symm = FLT_MAX;
	float4 min_q_a_symm;

	for (int i = 0; i < 24; i++) {
		float4 q_a_symm = hamilton (q_a, group_quat[i]);
		float dist_q_b_a_symm = distanceS (q_b, q_a_symm);
		if (dist_q_b_a_symm < min_dist_q_b_a_symm) {
			min_dist_q_b_a_symm = dist_q_b_a_symm;
			min_q_a_symm = q_a_symm;
		}
	}

	return min_q_a_symm;
}

__host__ __device__
float4 LocalSymmetryOptimization (const float4 group_quat[24],
                                  int group_quat_size,
                                  const float4 & q_a,
                                  const float4 & q_b) {
	float min_dist_q_b_a_symm = FLT_MAX;
	float4 min_q_a_symm;

	for (int i = 0; i < group_quat_size; i++) {
		float4 q_a_symm = hamilton (q_a, group_quat[i]);
		float dist_q_b_a_symm = distanceS (q_b, q_a_symm);
		if (dist_q_b_a_symm < min_dist_q_b_a_symm) {
			min_dist_q_b_a_symm = dist_q_b_a_symm;
			min_q_a_symm = q_a_symm;
		}
	}

	return min_q_a_symm;
}

__host__ __device__
void RelaxBiHarmonicQuaternions (const int3 & res3,
                                 const int3 & id3,
                                 const float4 group_quats[],
                                 bool use_group_symmetry,
                                 const float * bh_masks,
                                 const float * hc_quats,
                                 const float * w_hc_quats,
                                 const float * prev_quats,
                                 float * quats) {
	int id = id3.x + id3.y * res3.x + id3.z * res3.x * res3.y;
	int dim_mask = 5;

	// 1)	Compute the index of the relevant mask
	//		NB : the relaxation/smoothing mask depend on its evaluation
	//		location w.r.t the boundary of the domain (here a cube)
	int3 mask_id3;
	GridToMask (id3, res3, make_int3 (2, 2, 2), mask_id3);
	int mask_id = dim_mask * dim_mask * mask_id3.z + dim_mask * mask_id3.y + mask_id3.x;
	float4 prev_q = ((float4 *)prev_quats)[id];

	// 2) Compute the Linear Smoothing Operation
	float4 q_linear_operator = make_float4 (0.f, 0.f, 0.f, 0.f);
	for (int k_n = -2; k_n <= 2; k_n++)
		for (int j_n = -2; j_n <= 2; j_n++)
			for (int i_n = -2; i_n <= 2; i_n++) {
				int im = id3.x + i_n; int jm = id3.y + j_n; int km = id3.z + k_n;
				int id_neigh = res3.x * res3.y * km + res3.x * jm + im;
				int mask_id_neigh = dim_mask * dim_mask * (2 + k_n) + dim_mask * (2 + j_n) + (2 + i_n);
				float bh_mask_neigh_value = bh_masks[125 * mask_id + mask_id_neigh];
				if (bh_mask_neigh_value != 0.f) {
					float4 prev_q_neigh = ((float4 *) prev_quats)[id_neigh];
					if (use_group_symmetry)
						prev_q_neigh = LocalSymmetryOptimization (group_quats, 8,
						               prev_q_neigh, prev_q);
					q_linear_operator = q_linear_operator +
					                    bh_mask_neigh_value * prev_q_neigh;
				}
			}

	// 3)	Compute the residual term from
	// 		a) the value of the constraint
	// 		b) the weight of the constraint
	float4 q_hc = ((float4 *)hc_quats)[id];
	float lambda = w_hc_quats[id];
	if (use_group_symmetry)
		q_hc = LocalSymmetryOptimization (group_quats, 8, q_hc, prev_q);
	float4 residual = lambda * q_hc - (q_linear_operator + lambda * prev_q);

	// 4) Compute the relaxation
	float omega_relax = (1.f / 6.f);
	((float4 *) quats)[id] = prev_q + omega_relax * residual;
}

__global__
void RelaxBiHarmonicQuaternions (int3 res3, int3 pg_id3,
                                 int3 pg_res3,
                                 bool use_group_symmetry,
                                 const float * bh_masks,
                                 const float * hc_quats,
                                 const float * w_hc_quats,
                                 const float * prev_quats,
                                 float * quats) {

	int3 kernel_id3  = make_int3 (blockIdx.x * blockDim.x + threadIdx.x,
	                              blockIdx.y * blockDim.y + threadIdx.y,
	                              blockIdx.z * blockDim.z + threadIdx.z);

	int3 id3 = make_int3 (pg_res3.x * kernel_id3.x + pg_id3.x,
	                      pg_res3.y * kernel_id3.y + pg_id3.y,
	                      pg_res3.z * kernel_id3.z + pg_id3.z);

	if (id3.x >= res3.x || id3.y >= res3.y || id3.z >= res3.z)
		return;

	RelaxBiHarmonicQuaternions (res3, id3, gcube_quats, use_group_symmetry,
	                            bh_masks, hc_quats, w_hc_quats, prev_quats, quats);
}

__host__ __device__
void LaplacianSmoothQuaternions (const int3 & res3,
                                 const int3 & id3,
                                 const int3 & pg_id3,
                                 const float4 group_quats[],
                                 bool use_group_symmetry,
                                 const float * hc_quats,
                                 const float * w_hc_quats,
                                 const float * prev_quats,
                                 float * quats) {
	int id = id3.x + id3.y * res3.x + id3.z * res3.x * res3.y;
	int neigh_offs[6] = {1, -1, res3.x, -res3.x, res3.x * res3.y, -res3.x * res3.y};
	bool neigh_tests[6] = {(0 <= (id3.x + 1)) && ((id3.x + 1) < res3.x),
	                       (0 <= (id3.x - 1)) && ((id3.x - 1) < res3.x),
	                       (0 <= (id3.y + 1)) && ((id3.y + 1) < res3.y),
	                       (0 <= (id3.y - 1)) && ((id3.y - 1) < res3.y),
	                       (0 <= (id3.z + 1)) && ((id3.z + 1) < res3.z),
	                       (0 <= (id3.z - 1)) && ((id3.z - 1) < res3.z)
	                      };

	float4 neigh_prev_quat, quat_update, quat_id, quat_symm;
	float sum_w;

	sum_w = 0.f;
	quat_update = make_float4 (0.f, 0.f, 0.f, 0.f);
	quat_id = ((float4 *)prev_quats)[id];

	// 6-connex neighborhood
	for (int i = 0; i < 6; i++) {
		if (neigh_tests[i]) {
			neigh_prev_quat = ((float4 *)prev_quats)[id + neigh_offs[i]];
			if (fabs (neigh_prev_quat.x) < 10.f) {
				if (use_group_symmetry && fabs (quat_id.x) < 10.f)
					quat_symm = LocalSymmetryOptimization (group_quats, neigh_prev_quat, quat_id);
				else
					quat_symm = neigh_prev_quat;
				quat_update = quat_update + quat_symm;
				sum_w += 1.f;
			}
		}
	}

	// Add constraint by screening
	float4 hc_quat, hc_quat_symm;
	float w_hc_quat;
	hc_quat = ((float4 *)hc_quats)[id];
	w_hc_quat = w_hc_quats[id];
	if (w_hc_quat != 0.f) {
		if (use_group_symmetry && fabs (quat_id.x) < 10.f)
			hc_quat_symm = LocalSymmetryOptimization (group_quats, hc_quat, quat_id);
		else
			hc_quat_symm = hc_quat;
		quat_update = quat_update + w_hc_quat * hc_quat_symm;
		sum_w += w_hc_quat;
	}


	// Rescale according to weights
	quat_update = (1.f / sum_w) * quat_update;

	// Normalize quaternion
	quat_update = normalize (quat_update);

	if (sum_w == 0.f)
		((float4 *)quats)[id] = make_float4 (50.f, 50.f, 50.f, 50.f);
	else
		((float4 *)quats)[id] = quat_update;
}

__host__ __device__
void BiLaplacianSmoothQuaternions (const int3 & res3,
                                   const int3 & id3,
                                   const int3 & pg_id3,
                                   const float4 group_quats[],
                                   bool use_group_symmetry,
                                   const float * hc_quats,
                                   const float * w_hc_quats,
                                   const float * prev_quats,
                                   float * quats) {
	int id = id3.x + id3.y * res3.x + id3.z * res3.x * res3.y;
	int3 neigh_off[7];
	float4 neigh_prev_quat, quat_update, quat_id;
	//	float4 quat_symm;
	//	float sum_w;
	float lambda = 1.f;

	neigh_off[0] = make_int3 (0, 0, 0);
	neigh_off[1] = make_int3 (1, 0, 0);
	neigh_off[2] = make_int3 (0, 1, 0);
	neigh_off[3] = make_int3 (0, 0, 1);

	neigh_off[4] = make_int3 (-1, 0, 0);
	neigh_off[5] = make_int3 (0, -1, 0);
	neigh_off[6] = make_int3 (0, 0, -1);


	//	sum_w = 0.f;
	quat_update = make_float4 (0.f, 0.f, 0.f, 0.f);
	quat_id = ((float4 *)prev_quats)[id];

	bool debug = (id3.x == 9) && (id3.y == 9) && (id3.z == 4);

	if (debug)
		printf ("%d %d %d\n", id3.x, id3.y, id3.z);

	if (debug)
		printf ("quat_id : %f %f %f %f\n",
		        quat_id.x, quat_id.y, quat_id.z, quat_id.w);

	// 6-connex BiLaplacian
	// L^T*L = D + P
	// Compute L^T*L*X and D
	float4 ll_quat_update = make_float4 (0, 0, 0, 0);

	// 1) acum weights
	float sum_w_i = 0.f;
	for (int i = 1; i <= 6; i++) {
		int3 id3_off_i = id3 + neigh_off[i];
		if (TestLimit (id3_off_i, res3))
			sum_w_i += 1.f;
	}

	if (debug)
		printf ("sum_w_i : %f\n", sum_w_i);

	// 2) compute R*L*X = R*(R - I)*X
	float w_diag_i = 0.f;
	float inv_sum_w_i = (1.f / sum_w_i);
	for (int i = 0; i <= 6; i++) {
		int3 id3_off_i = id3 + neigh_off[i];
		int id_off_i = id3_off_i.x + id3_off_i.y * res3.x + id3_off_i.z * res3.x * res3.y;
		float4 quat_update_i = make_float4 (0, 0, 0, 0);

		if (TestLimit (id3_off_i, res3)) {
			// Compute L*X = (R - I)*X
			// 2.a) acum weights
			float sum_w_i_j = 0.f;
			float w_diag_i_j = 0.f;
			for (int j = 1; j <= 6; j++) {
				int3 id3_off_i_j = id3_off_i + neigh_off[j];
				if (TestLimit (id3_off_i_j, res3))
					sum_w_i_j += 1.f;
			}

			if (debug)
				printf ("sum_w_i_j : %f\n", sum_w_i_j);

			// 2.b) compute R*X
			float inv_sum_w_i_j = (1.f / sum_w_i_j);
			for (int j = 1; j <= 6; j++) {
				int3 id3_off_i_j = id3_off_i + neigh_off[j];
				if (TestLimit (id3_off_i_j, res3)) {
					int id_off_i_j = id3_off_i_j.x + id3_off_i_j.y * res3.x + id3_off_i_j.z * res3.x * res3.y;
					neigh_prev_quat = ((float4 *)prev_quats)[id_off_i_j];
					quat_update_i = quat_update_i + inv_sum_w_i_j * neigh_prev_quat;
					if ((id3_off_i_j.x == id3.x) && (id3_off_i_j.y == id3.y) && (id3_off_i_j.z == id3.z))
						w_diag_i_j += inv_sum_w_i_j;
				}
			}

			if (debug)
				printf ("R : quat_update_i : %f %f %f %f\n",
				        quat_update_i.x, quat_update_i.y, quat_update_i.z, quat_update_i.w);

			// 2.c) compute (R - I)*X
			quat_update_i = quat_update_i - ((float4 *)prev_quats)[id_off_i];

			if (debug)
				printf ("R - I : quat_update_i : %f %f %f %f\n",
				        quat_update_i.x, quat_update_i.y, quat_update_i.z, quat_update_i.w);

			// 2.d) compute R*(R - I)*X
			if (i == 0) {
				ll_quat_update = (-1.f) * quat_update_i;
				w_diag_i += 1.f;
				//				ll_quat_update = quat_update_i;
				//				w_diag_i += (-1.f);
			} else {
				ll_quat_update = ll_quat_update + inv_sum_w_i * quat_update_i;
				w_diag_i += inv_sum_w_i * w_diag_i_j;
			}
		}

		//		if (debug)
		//			printf ("w_diag_i : %f\n", w_diag_i);
	}

	//	// 3) compute L^T*L*X = (R - I)*L*X
	//	ll_quat_update = ll_quat_update - quat_id;
	//
	//	// 4) compute D and P*X = L^T*L*X - D*X
	//	float diag = w_diag_i - 1.f;
	float diag = w_diag_i;
	float4 p_quat_update = ll_quat_update - diag * quat_id;

	// 5) compute (D + lambda)^[-1]*(lambda*B - R*X)
	float4 hc_quat;
	float w_hc_quat;
	hc_quat = ((float4 *)hc_quats)[id];
	w_hc_quat = w_hc_quats[id];
	if (w_hc_quat != 0.f && false) {
		quat_update = (1.f / (diag + lambda)) * (lambda * hc_quat - p_quat_update);
	} else {
		//		quat_update = -(1.f/(diag))*p_quat_update;
		quat_update = ll_quat_update;
	}

	if (debug) {
		printf ("q[%d][%d][%d] : %f %f %f %f\n",
		        id3.x, id3.y, id3.z,
		        quat_update.x, quat_update.y, quat_update.z, quat_update.w);
		printf ("diag : %f\n", diag);
		printf ("w_hc_quat : %f\n", w_hc_quat);
		printf ("hc_quat : %f %f %f %f\n", hc_quat.x, hc_quat.y, hc_quat.z, hc_quat.w);
		printf ("ll_quat_update : %f %f %f %f\n", ll_quat_update.x, ll_quat_update.y, ll_quat_update.z, ll_quat_update.w);
	}

	// Normalize quaternion
	//	quat_update = normalize (quat_update);

	((float4 *)quats)[id] = quat_update;
}

__host__ __device__
void LaplacianSmoothQuaternionsError (const int3 & res3,
                                      const int3 & id3,
                                      const int3 & pg_id3,
                                      const float4 group_quats[],
                                      bool use_group_symmetry,
                                      const float * hc_quats,
                                      const float * w_hc_quats,
                                      const float * quats,
                                      float * error) {
	int id = id3.x + id3.y * res3.x + id3.z * res3.x * res3.y;
	int neigh_offs[3] = {1, res3.x, res3.x * res3.y};
	bool neigh_tests[3] = {(0 <= (id3.x + 1)) && ((id3.x + 1) < res3.x),
	                       (0 <= (id3.y + 1)) && ((id3.y + 1) < res3.y),
	                       (0 <= (id3.z + 1)) && ((id3.z + 1) < res3.z)
	                      };

	float4 neigh_quat, quat_id, quat_symm;
	float sum_error;

	quat_id = ((float4 *)quats)[id];

	sum_error = 0.f;

	// Compute Error for the three (left, right, up)
	// edges from a 6-connex neighborhood
	for (int i = 0; i < 3; i++) {
		if (neigh_tests[i]) {
			neigh_quat = ((float4 *)quats)[id + neigh_offs[i]];
			if (use_group_symmetry)
				quat_symm = LocalSymmetryOptimization (group_quats, neigh_quat, quat_id);
			else
				quat_symm = neigh_quat;
			sum_error += distanceS (quat_symm, quat_id);
		}
	}

	// Compute screening constraint error
	float4 hc_quat, hc_quat_symm;
	float w_hc_quat;
	hc_quat = ((float4 *)hc_quats)[id];
	w_hc_quat = w_hc_quats[id];
	if (w_hc_quat != 0.f) {
		if (use_group_symmetry)
			hc_quat_symm = LocalSymmetryOptimization (group_quats, hc_quat, quat_id);
		else
			hc_quat_symm = hc_quat;
		sum_error += w_hc_quat * distanceS (hc_quat_symm, quat_id);
	}

	error[id] = sum_error;
}

__global__ void LaplacianSmoothQuaternions (int3 res3, int3 pg_id3,
        bool use_group_symmetry,
        const float * hc_quats,
        const float * w_hc_quats,
        const float * prev_quats,
        float * quats) {

	int3 kernel_id3  = make_int3 (blockIdx.x * blockDim.x + threadIdx.x,
	                              blockIdx.y * blockDim.y + threadIdx.y,
	                              blockIdx.z * blockDim.z + threadIdx.z);

	int3 id3 = make_int3 (2 * kernel_id3.x + pg_id3.x,
	                      2 * kernel_id3.y + pg_id3.y,
	                      2 * kernel_id3.z + pg_id3.z);

	if (id3.x >= res3.x || id3.y >= res3.y || id3.z >= res3.z)
		return;

	LaplacianSmoothQuaternions (res3, id3, pg_id3, gcube_quats, use_group_symmetry,
	                            hc_quats, w_hc_quats, prev_quats, quats);
}

__global__
void BiLaplacianSmoothQuaternions (int3 res3, int3 pg_id3,
                                   bool use_group_symmetry,
                                   const float * hc_quats,
                                   const float * w_hc_quats,
                                   const float * prev_quats,
                                   float * quats) {

	int3 kernel_id3  = make_int3 (blockIdx.x * blockDim.x + threadIdx.x,
	                              blockIdx.y * blockDim.y + threadIdx.y,
	                              blockIdx.z * blockDim.z + threadIdx.z);

	int3 id3 = make_int3 (kernel_id3.x + pg_id3.x,
	                      kernel_id3.y + pg_id3.y,
	                      kernel_id3.z + pg_id3.z);

	if (id3.x >= res3.x || id3.y >= res3.y || id3.z >= res3.z)
		return;

	BiLaplacianSmoothQuaternions (res3, id3, pg_id3, gcube_quats, use_group_symmetry,
	                              hc_quats, w_hc_quats, prev_quats, quats);
}

__global__
void LaplacianSmoothQuaternionsError (int3 res3, int3 pg_id3,
                                      bool use_group_symmetry,
                                      const float * hc_quats,
                                      const float * w_hc_quats,
                                      const float * quats,
                                      float * error) {

	int3 kernel_id3  = make_int3 (blockIdx.x * blockDim.x + threadIdx.x,
	                              blockIdx.y * blockDim.y + threadIdx.y,
	                              blockIdx.z * blockDim.z + threadIdx.z);

	int3 id3 = make_int3 (2 * kernel_id3.x + pg_id3.x,
	                      2 * kernel_id3.y + pg_id3.y,
	                      2 * kernel_id3.z + pg_id3.z);

	if (id3.x >= res3.x || id3.y >= res3.y || id3.z >= res3.z)
		return;

	LaplacianSmoothQuaternionsError (res3, id3, pg_id3, gcube_quats, use_group_symmetry,
	                                 hc_quats, w_hc_quats, quats, error);
}

__global__
void QuaternionsInterpolation (const float4 * quats_up,
                               const int3 res3_up, const int3 res3_down,
                               float4 * quats_down) {

	int3 id3_down  = make_int3 (blockIdx.x * blockDim.x + threadIdx.x,
	                            blockIdx.y * blockDim.y + threadIdx.y,
	                            blockIdx.z * blockDim.z + threadIdx.z);

	if (id3_down.x >= res3_down.x || id3_down.y >= res3_down.y || id3_down.z >= res3_down.z)
		return;

	int3 id3_down_up = make_int3 (id3_down.x * (res3_down.x / res3_up.x),
	                              id3_down.y * (res3_down.y / res3_up.y),
	                              id3_down.z * (res3_down.z / res3_up.z));
	int id_down_up = id3_down_up.x + id3_down_up.y * res3_up.x
	                 + id3_down_up.z * (res3_up.x * res3_up.y);

	int id_down = id3_down.x + id3_down.y * res3_down.x
	              + id3_down.z * (res3_down.x * res3_down.y);

	float4 quat_interpol;
	quat_interpol = quats_up [id_down_up];

	quats_down[id_down] = quat_interpol;
}

void FrameField::SolveLocalOptimization (int num_iters, bool use_group_symmetry) {
	double time1, time2;
	int3 res3, pg_res3;
	dim3 block_dim, grid_dim;

	// Set CUDA kernel launch config
	pg_res3 = make_int3 (3, 3, 3);
	res3 = make_int3 (res_[0], res_[1], res_[2]);
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 ((res_[0] / (pg_res3.x * block_dim.x)) + 1,
	                 (res_[1] / (pg_res3.y * block_dim.y)) + 1,
	                 (res_[2] / (pg_res3.z * block_dim.z)) + 1);


	// Set (GPU and GPU) Quaternion Weighted Constraints
	bool init_quats_from_first_constraint = false;

	for (int c = 0; c < (int)hc_points_.size (); c++) {
		Vec3f wc_pos = hc_points_[c].position ();
		Vec3i wc_ipos (wc_pos[0], wc_pos[1], wc_pos[2]);
		Vec3f wc_twist = hc_points_[c].twist ();
		int wc_id = wc_ipos[0] + res_[0] * wc_ipos[1] + res_[0] * res_[1] * wc_ipos[2];
		float wc_theta = wc_twist.length ();
		Vec3f wc_axis = normalize (wc_twist);
		float4 hc_quat;
		float w_hc_quat = 1.f;
		hc_quat.x = std::sin (0.5 * wc_theta) * wc_axis[0];
		hc_quat.y = std::sin (0.5 * wc_theta) * wc_axis[1];
		hc_quat.z = std::sin (0.5 * wc_theta) * wc_axis[2];
		hc_quat.w = std::cos (0.5 * wc_theta);

		std::cout << "wc_hc_quat : "
		          << hc_quat.x << ", " << hc_quat.y << ", " << hc_quat.z << ", " << hc_quat.w
		          << " | " << wc_twist
		          << " | " << wc_ipos
		          << " | " << wc_pos
		          << std::endl;

		if (c != 0 || !init_quats_from_first_constraint) {
			// GPU
			cudaMemcpy (hc_quats_ + 4 * wc_id, &hc_quat.x, sizeof (float4),
			            cudaMemcpyHostToDevice);
			CheckCUDAError ();
			cudaMemcpy (w_hc_quats_ + wc_id, &w_hc_quat, sizeof (float),
			            cudaMemcpyHostToDevice);
			CheckCUDAError ();

			// CPU
			hc_quats_host_[wc_id] = hc_quat;
			w_hc_quats_host_[wc_id] = w_hc_quat;
		} else {
			// Initialize CPU Data to the first constraint
			for (int i = 0; i < res_[0]; i++)
				for (int j = 0; j < res_[1]; j++)
					for (int k = 0; k < res_[2]; k++) {
						quat_field_ping_[i + res_[0]*j + res_[0]*res_[1]*k] = hc_quat;
					}

			cudaMemcpy (init_quats_, &(quat_field_ping_[0].x),
			            res_[0]*res_[1]*res_[2]*sizeof (float4),
			            cudaMemcpyHostToDevice);
			CheckCUDAError ();
		}
	}

	// Initialize Quaternion buffers
	cudaMemcpy (quats_ping_, init_quats_,
	            res_[0]*res_[1]*res_[2]*sizeof (float4),
	            cudaMemcpyDeviceToDevice);
	CheckCUDAError ();

	cudaMemcpy (&(quat_field_pong_[0].x), init_quats_,
	            res_[0]*res_[1]*res_[2]*sizeof (float4),
	            cudaMemcpyDeviceToHost);
	CheckCUDAError ();

	float * quats_ping = quats_ping_;
	float * quats_pong = quats_pong_;

	// Allocate Error Metric Data
	thrust::device_vector<float> quat_error (res_[0]*res_[1]*res_[2]);

	/*
	 * Iterate Local Non Linear Optimization
	 */
	time1 = GET_TIME ();
	for (int iter = 0; iter < num_iters; iter++) {
		// CUDA Profiling only on last iteration
		//		if (iter == (num_iters - 1))
		//			cudaProfilerStart ();
		for (int pg_x = 0; pg_x < pg_res3.x; pg_x++)
			for (int pg_y = 0; pg_y < pg_res3.y; pg_y++)
				for (int pg_z = 0; pg_z < pg_res3.z; pg_z++) {
					int3 pg_id3 = make_int3 (pg_x, pg_y, pg_z);
					//					LaplacianSmoothQuaternions<<<grid_dim, block_dim>>>
					//						(res3,
					//						 pg_id3,
					//						 use_group_symmetry,
					//						 hc_quats_,
					//						 w_hc_quats_,
					//						 quats_ping,
					//						 quats_pong);

					//					BiLaplacianSmoothQuaternions<<<grid_dim, block_dim>>>
					//						(res3,
					//						 pg_id3,
					//						 use_group_symmetry,
					//						 hc_quats_,
					//						 w_hc_quats_,
					//						 quats_ping,
					//						 quats_pong);

					RelaxBiHarmonicQuaternions <<< grid_dim, block_dim>>>
					(res3,
					 pg_id3,
					 pg_res3,
					 use_group_symmetry,
					 bh_masks_,
					 hc_quats_,
					 w_hc_quats_,
					 quats_ping,
					 quats_pong);
				}
		cudaDeviceSynchronize ();

		// Swap Buffers
		std::swap (quats_ping, quats_pong);

	}
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << std::endl;
	std::cout << "Frame Field Generation in " << time2 - time1 << "ms on GPU" << std::endl;

	// Final swap and get back GPU data
	if ((num_iters % 2) == 1)
		std::swap (quats_ping, quats_pong);

	cudaMemcpy (&(quat_field_ping_[0].x), quats_pong,
	            quat_field_ping_.size ()*sizeof (float4),
	            cudaMemcpyDeviceToHost);
	CheckCUDAError ();

	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	for (int i = 0; i < dim_x; i++)
		for (int j = 0; j < dim_y; j++)
			for (int k = 0; k < dim_z; k++) {
				if ((i == 8) && (j == 9) && (k == 4)) {
					for (int k_neigh = k - 4; k_neigh <= k + 4; k_neigh++) {
						for (int j_neigh = j - 4; j_neigh <= j + 4; j_neigh++) {
							for (int i_neigh = i - 4; i_neigh <= i + 4; i_neigh++) {
								float4 value = quat_field_ping_[i_neigh + dim_x * j_neigh + dim_x * dim_y * k_neigh];
								std::cout << value.y << " ";
							}
							std::cout << std::endl;
						}
						std::cout << std::endl;
					}
				}
			}

	std::cout << "L^T*L Matrix Test" << std::endl;

	for (int i = 0; i < (int)twist_field_ping_.size (); i++) {
		float4 quat = quat_field_ping_[i];
		quat = normalize (quat);
		Vec3f quat_xyz (quat.x, quat.y, quat.z);
		float quat_w = quat.w;
		float theta = 2.0 * std::atan2 (quat_xyz.length (), quat_w);
		twist_field_ping_[i] = theta * normalize (quat_xyz);
	}
}


