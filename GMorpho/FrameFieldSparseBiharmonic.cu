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

#include "FrameField.h"

using namespace MorphoGraphics;

void FrameField::InitSparseBiharmonicSystem (const Vec3f & bbox_min, 
																						 const Vec3f & bbox_max, 
																						 const Vec3i & res) {
	bbox_min_ = bbox_min;
	bbox_max_ = bbox_max;
	res_ = res;
	cell_size_ = max (fabs (bbox_max_[0] - bbox_min_[0]), 
										max (fabs (bbox_max_[1] - bbox_min_[1]), 
												 fabs (bbox_max_[2] - bbox_min_[2])
												)
									 )/res_[0];

	slice_xy_id_ = 0;
	max_hc_number_ = 20;
	cu_csr_schol_p_buffer_ = NULL;
	cu_csr_schol_info_ = NULL;
	cusolver_handle_ = NULL;
	cusparse_handle_ = NULL;
	cu_stream_ = NULL;
	descr_bh_ = NULL;

	x_bh_ = NULL;
	rhs_bh_ = NULL;
	csrv_bh_ = NULL;
	csrc_bh_ = NULL;
	csrr_bh_ = NULL;
	nnz_bh_ = 0;
	csrv_ltl_ = NULL;
	csrc_ltl_ = NULL;
	csrr_ltl_ = NULL;
	nnz_ltl_ = 0;
	csrv_s_ = NULL;
	csrc_s_ = NULL;
	csrr_s_ = NULL;
	nnz_s_ = 0;

	quats_pong_ = NULL;
	quats_ping_ = NULL;

	lambda_ = 10.f;

	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	int num_cells = dim_x*dim_y*dim_z;
	int num_hcs = hc_points_.size ();

	twist_field_ping_.resize (num_cells);
	position_field_ping_.resize (num_cells);

	float dx = cell_size_;
	float scale = dx;

	opengl_frame_field_p_.clear ();
	opengl_frame_field_n_.clear ();

	for (int i = 0; i < dim_x; i++)
		for (int j = 0; j < dim_y; j++)
			for (int k = 0; k < dim_z; k++) {
				Vec3f pos = scale*Vec3f (i, j, k) 
					+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]) 
					+ 0.5f*Vec3f (dx, dx, dx);
				position_field_ping_[i + dim_x*j + dim_x*dim_y*k] = pos;
				twist_field_ping_[i + dim_x*j + dim_x*dim_y*k] = Vec3f (0, 0, 0);
			}

	num_tris_in_cube_ = 12;	
	opengl_frame_field_p_.reserve (3*num_cells*num_tris_in_cube_);
	opengl_frame_field_n_.reserve (3*num_cells*num_tris_in_cube_);
	ComputeOpenGLCube ();

	int m_range = 2;
	int m_dim = 2*m_range + 1;
	int max_nnz = m_dim*m_dim*m_dim*num_cells;
	n_bh_ = num_cells;

	// Initialize CUSolver and CUSparse Libraries
	cusparseStatus_t cusparse_status;
	cusolverStatus_t cusolver_status;
	cusolver_status = cusolverSpCreate (&cusolver_handle_);
	if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
		std::cout << "CUSolver Handle Creation Failed" << std::endl;
	}
	CheckCUDAError ();
	cusparse_status = cusparseCreate (&cusparse_handle_);
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "CUSparse Handle Creation Failed" << std::endl;
	}
	CheckCUDAError ();

	cudaStreamCreate (&cu_stream_);
	CheckCUDAError ();
	cusolverSpSetStream (cusolver_handle_, cu_stream_);
	CheckCUDAError ();
	cusparseSetStream (cusparse_handle_, cu_stream_);
	CheckCUDAError ();

	// Allocate GPU Resources
	cusparseCreateMatDescr (&descr_bh_);
	CheckCUDAError ();
	cusparseSetMatType (descr_bh_, CUSPARSE_MATRIX_TYPE_GENERAL);
	CheckCUDAError ();
	cusparseSetMatIndexBase (descr_bh_, CUSPARSE_INDEX_BASE_ZERO);
	CheckCUDAError ();

	cudaMalloc (&x_bh_, n_bh_*sizeof (float));
	CheckCUDAError ();
	cudaMalloc (&rhs_bh_, n_bh_*sizeof (float));
	CheckCUDAError ();

	// Allocate L^T*L Sparse Matrix
	cudaMalloc (&csrv_ltl_, max_nnz*sizeof (float));
	CheckCUDAError ();
	cudaMalloc (&csrc_ltl_, max_nnz*sizeof (int));
	CheckCUDAError ();
	cudaMalloc (&csrr_ltl_, (n_bh_ + 1)*sizeof (int));
	CheckCUDAError ();

	// Allocate S Sparse Matrix (Screen term)
	cudaMalloc (&csrv_s_, n_bh_*sizeof (float));
	CheckCUDAError ();
	cudaMalloc (&csrc_s_, n_bh_*sizeof (int));
	CheckCUDAError ();
	cudaMalloc (&csrr_s_, (n_bh_ + 1)*sizeof (int));
	CheckCUDAError ();

	// Allocate Full L^T*L + S Sparse Matrix
	cudaMalloc (&csrv_bh_, max_nnz*sizeof (float));
	CheckCUDAError ();
	cudaMalloc (&csrc_bh_, max_nnz*sizeof (int));
	CheckCUDAError ();
	cudaMalloc (&csrr_bh_, (n_bh_ + 1)*sizeof (int));
	CheckCUDAError ();

	BuildSparseLTLMatrix ();
	BuildSparseSMatrix ();

	// Allocate Quaternion buffers
	cudaMalloc (&quats_pong_, num_cells*sizeof (float4));
	CheckCUDAError ();
	cudaMalloc (&quats_ping_, num_cells*sizeof (float4));
	CheckCUDAError ();
}

void FrameField::BuildSparseLTLMatrix () {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];

	// Construct (CPU) CSR Sparse Matrix: 
	// Laplacian	
	std::vector<float> host_csrv;
	std::vector<int> host_csrc;
	std::vector<int> host_csrr;

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

	host_csrr.push_back (host_csrv.size ());
	for (int k = 0; k < dim_z; k++)
		for (int j = 0; j < dim_y; j++) 
			for (int i = 0; i < dim_x; i++) {
				int cell_id = dim_y*dim_x*k + dim_x*j + i;
				float sum_mask = 0.f;
				for (int l = 0; l < (int)mask_vals.size (); l++) {
					int im = i + mask_ids[l][0];
					int jm = j + mask_ids[l][1];
					int km = k + mask_ids[l][2];
					if ((0 <= im) && (im < dim_x) &&
							(0 <= jm) && (jm < dim_y) &&
							(0 <= km) && (km < dim_z)) {
						sum_mask += mask_vals[l];
					}
				}

				//				sum_mask = 6.f;
				for (int l = 0; l < (int)mask_vals.size (); l++) {
					int im = i + mask_ids[l][0];
					int jm = j + mask_ids[l][1];
					int km = k + mask_ids[l][2];
					int mask_cell_id = dim_y*dim_x*km + dim_x*jm + im;
					if ((0 <= im) && (im < dim_x) &&
							(0 <= jm) && (jm < dim_y) &&
							(0 <= km) && (km < dim_z)) {
						host_csrc.push_back (mask_cell_id);
						float mask_val;
						if (cell_id != mask_cell_id)
							mask_val = -(mask_vals[l]/sum_mask);
						else {
							mask_val = 1.f - (mask_vals[l]/sum_mask);
						}
						host_csrv.push_back (mask_val);
					}
				}

				host_csrr.push_back (host_csrv.size ());
			}

	nnz_l_ = (int)host_csrv.size ();

	csrv_l_ = NULL;
	csrc_l_ = NULL;
	csrr_l_ = NULL;
	int n_l = n_bh_;

	cudaMalloc (&csrv_l_, nnz_l_*sizeof (float));
	CheckCUDAError ();
	cudaMalloc (&csrc_l_, nnz_l_*sizeof (int));
	CheckCUDAError ();
	cudaMalloc (&csrr_l_, (n_l + 1)*sizeof (int));
	CheckCUDAError ();

	// Copy the CSR Sparse System to GPU
	cudaMemcpy (csrv_l_, &host_csrv[0], nnz_l_*sizeof (float), 
							cudaMemcpyHostToDevice);
	CheckCUDAError ();
	cudaMemcpy (csrc_l_, &host_csrc[0], nnz_l_*sizeof (int), 
							cudaMemcpyHostToDevice);
	CheckCUDAError ();
	cudaMemcpy (csrr_l_, &host_csrr[0], (n_l + 1)*sizeof (int), 
							cudaMemcpyHostToDevice);
	CheckCUDAError ();

	// Compute L^t*L on the GPU
	cusparseStatus_t cusparse_status;
	int opa_num_r = n_l;
	int opa_num_c = n_l;
	//	int opb_num_r = n_l;
	int opb_num_c = n_l;
	int base_ltl;
	int * host_nnz_total = &nnz_ltl_;
	base_ltl = 0;
	cusparseSetPointerMode (cusparse_handle_, CUSPARSE_POINTER_MODE_HOST);
	//	cudaMalloc ((void**) &csrr_ls_, (opa_num_r + 1)*sizeof (int));
	cusparse_status = cusparseXcsrgemmNnz (cusparse_handle_, 
																				 CUSPARSE_OPERATION_TRANSPOSE,  
																				 CUSPARSE_OPERATION_NON_TRANSPOSE,  
																				 opa_num_r, opb_num_c, opa_num_c, 
																				 descr_bh_, nnz_l_, csrr_l_, csrc_l_,
																				 descr_bh_, nnz_l_, csrr_l_, csrc_l_,
																				 descr_bh_, csrr_ltl_, host_nnz_total);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "CUSparse Operation failed." << std::endl;
	}	
	
	if (NULL != host_nnz_total) {
		nnz_ltl_ = *host_nnz_total;
	} else {
		cudaMemcpy (&nnz_ltl_, csrr_ltl_ + opa_num_r, sizeof (int), cudaMemcpyDeviceToHost);
		cudaMemcpy (&base_ltl, csrr_ltl_, sizeof (int), cudaMemcpyDeviceToHost);
		nnz_ltl_ -= base_ltl;
	}
	//	cudaMalloc ((void**)&csrc_ls_, nnz_ls_*sizeof (int));
	//	cudaMalloc ((void**)&csrv_ls_, nnz_ls_*sizeof (float));
	cusparse_status = cusparseScsrgemm (cusparse_handle_, 
																			CUSPARSE_OPERATION_TRANSPOSE,  
																			CUSPARSE_OPERATION_NON_TRANSPOSE,  
																			opa_num_r, opb_num_c, opa_num_c, 
																			descr_bh_, nnz_l_, csrv_l_, csrr_l_, csrc_l_,
																			descr_bh_, nnz_l_, csrv_l_, csrr_l_, csrc_l_,
																			descr_bh_, csrv_ltl_, csrr_ltl_, csrc_ltl_);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "CUSparse Operation failed." << std::endl;
	}	
	//	cudaFree (csrv_l);
	//	cudaFree (csrr_l);
	//	cudaFree (csrc_l);
}

void FrameField::BuildSparseSMatrix () {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];

	// Construct (CPU) CSR Sparse Matrix: 
	// Laplacian	
	std::vector<float> host_csrv;
	std::vector<int> host_csrc;
	std::vector<int> host_csrr;

	host_csrr.push_back (host_csrv.size ());
	for (int k = 0; k < dim_z; k++)
		for (int j = 0; j < dim_y; j++) 
			for (int i = 0; i < dim_x; i++) {
				int cell_id = dim_y*dim_x*k + dim_x*j + i;
				host_csrc.push_back (cell_id);
				host_csrv.push_back (1);
				host_csrr.push_back (host_csrv.size ());
			}

	//	host_csrr.push_back (host_csrv.size ());
	nnz_s_ = (int)host_csrv.size ();

	// Copy the CSR Sparse System to GPU
	cudaMemcpy (csrv_s_, &host_csrv[0], nnz_s_*sizeof (float), 
							cudaMemcpyHostToDevice);
	CheckCUDAError ();
	cudaMemcpy (csrc_s_, &host_csrc[0], nnz_s_*sizeof (int), 
							cudaMemcpyHostToDevice);
	CheckCUDAError ();
	cudaMemcpy (csrr_s_, &host_csrr[0], (n_bh_ + 1)*sizeof (int), 
							cudaMemcpyHostToDevice);
	CheckCUDAError ();
}

void FrameField::BuildSparseBiharmonicSystem () {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	std::vector<float> host_csrv;
	std::vector<int> host_csrc;
	std::vector<int> host_csrr;
	cusparseStatus_t cusparse_status;
	// Set (GPU) Screen Term
	cudaMemset (csrv_s_, 0, n_bh_*sizeof (float));
	for (int c = 0; c < (int)hc_points_.size (); c++) {
		Vec3f wc_pos = hc_points_[c].position ();
		Vec3i wc_ipos (wc_pos[0], wc_pos[1], wc_pos[2]);
		int wc_id = wc_ipos[0] + dim_x*wc_ipos[1] + dim_x*dim_y*wc_ipos[2];
		float wc_val = 1.f; // TODO: lambda based soft constraint !!!!

		cudaMemcpy (csrv_s_ + wc_id, &wc_val, sizeof (float), 
								cudaMemcpyHostToDevice);
		CheckCUDAError ();
	}

	float alpha = 1.f, beta = 1.f;
	int base_bh;
	int * host_nnz_bh_total = &nnz_bh_;
	cusparseSetPointerMode (cusparse_handle_, CUSPARSE_POINTER_MODE_HOST);

	cusparseXcsrgeamNnz (cusparse_handle_, n_bh_, n_bh_, 
											 descr_bh_, nnz_ltl_, csrr_ltl_, csrc_ltl_, 
											 descr_bh_, nnz_s_, csrr_s_, csrc_s_, 
											 descr_bh_, csrr_bh_, host_nnz_bh_total);

	if (NULL != host_nnz_bh_total) {
		nnz_bh_ = *host_nnz_bh_total;
	} else {
		cudaMemcpy (&nnz_bh_, csrr_bh_ + n_bh_, sizeof (int), cudaMemcpyDeviceToHost);
		cudaMemcpy (&base_bh, csrr_bh_, sizeof (int), cudaMemcpyDeviceToHost);
		nnz_bh_ -= base_bh;
	}

	cusparse_status = cusparseScsrgeam (cusparse_handle_, n_bh_, n_bh_, 
																			&alpha, 
																			descr_bh_, 
																			nnz_ltl_, csrv_ltl_, csrr_ltl_, csrc_ltl_, 
																			&beta, 
																			descr_bh_, 
																			nnz_s_, csrv_s_, csrr_s_, csrc_s_, 
																			descr_bh_, 
																			csrv_bh_, csrr_bh_, csrc_bh_);

	cudaDeviceSynchronize ();
	CheckCUDAError ();
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "CUSparse Operation failed." << std::endl;
	} 	
	//	host_csrv.resize (nnz_bh_);
	//	host_csrc.resize (nnz_bh_);
	//	host_csrr.resize (n_bh_ + 1);
	//	// Copy the CSR Sparse System to CPU
	//	cudaMemcpy (&host_csrv[0], csrv_bh_, nnz_bh_*sizeof (float), 
	//							cudaMemcpyDeviceToHost);
	//	CheckCUDAError ();
	//	cudaMemcpy (&host_csrc[0], csrc_bh_, nnz_bh_*sizeof (int), 
	//							cudaMemcpyDeviceToHost);
	//	CheckCUDAError ();
	//	cudaMemcpy (&host_csrr[0], csrr_bh_, (n_bh_ + 1)*sizeof (int), 
	//							cudaMemcpyDeviceToHost);
	//	CheckCUDAError ();
	//
	//	int test_row = 6443;
	//	for (int check_row = test_row; check_row < (test_row + 1); check_row++) {
	//		int start_id = host_csrr[check_row];
	//		int end_id = host_csrr[check_row + 1];
	//		std::cout << "row : " << check_row << std::endl;
	//		for (int i = start_id; i < end_id; i++) {
	//			std::cout << "host_csrv[" << i << "] : " << host_csrv[i] << std::endl;
	//		}
	//		std::cout << std::endl;
	//	}
}

__global__ void CopyStride (float * dst, float * src, 
														int dst_stride_id, int dst_stride_size, int size) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= size)
		return;

	dst[dst_stride_size*id + dst_stride_id] = src[id];

}

void FrameField::StrideGPUCopy (float * dst, float * src, 
																int dst_stride_id, int dst_stride_size, int size) {

	int block_dim = 512;
	int grid_dim = (size/block_dim) + 1;
	CopyStride<<<grid_dim, block_dim>>> 
						(dst, src, dst_stride_id, dst_stride_size, size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
}

void FrameField::SolveSparseBiharmonicSystem (bool update_only) {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];

	cusolverStatus_t cusolver_status;

	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	float tol = 1.e-12;
	//	int reorder = 0;
	//	int rank = 0;
	//	int singularity = 0;
	//	float min_norm = 0.f;
	float * sys_csr_vals = NULL;
	int * sys_csr_row_inds = NULL;
	int * sys_csr_col_inds = NULL;
	float * sys_rhs = NULL;
	float * sys_x = NULL;
	//	int * sys_perm = NULL;
	int sys_nnz = 0;
	int sys_n = 0;
	std::vector<float> host_x;


	sys_csr_vals = csrv_bh_;
	sys_csr_row_inds = csrr_bh_;
	sys_csr_col_inds = csrc_bh_;
	sys_x = x_bh_;
	sys_rhs = rhs_bh_;
	sys_nnz = nnz_bh_;

	//	sys_csr_vals = csrv_ls_;
	//	sys_csr_row_inds = csrr_ls_;
	//	sys_csr_col_inds = csrc_ls_;
	//	sys_x = x_bh_;
	//	sys_rhs = rhs_bh_;
	//	sys_nnz = nnz_ls_;

	sys_n = n_bh_;
	host_x.resize (n_bh_);

	std::vector<float> rot_field (9*n_bh_);
	std::vector<float> dir_field (3*n_bh_);
	std::vector<float> complex_field (2*n_bh_);
	std::vector<float> q_field (4*n_bh_);

	for (int k = 0; k < 4; k++) {
		// Initialize (GPU) Right Hand Side vector
		cudaMemset (sys_rhs, 0, sys_n*sizeof (float));
		CheckCUDAError ();

		// Set (GPU) Right Hand Side vector
		// Weighted Constraints
		for (int c = 0; c < (int)hc_points_.size (); c++) {
			Vec3f wc_pos = hc_points_[c].position ();
			Vec3i wc_ipos (wc_pos[0], wc_pos[1], wc_pos[2]);
			Vec3f wc_twist = hc_points_[c].twist ();
			int wc_id = wc_ipos[0] + dim_x*wc_ipos[1] + dim_x*dim_y*wc_ipos[2];
			//			float wc_rhs_val = 10.f*wc_twist[k];
			//			float wc_rhs_val = wc_twist[k];
			Vec3f e_x (1, 0, 0);
			Vec3f e_y (0, 1, 0);
			Vec3f e_z (0, 0, 1);
			//			float wc_r[3][3];
			//				Vec3f v = 4.f*wc_twist;
			Vec3f v = wc_twist;
			//			std::cout << "wc_twist[" << c << "] : " << v << std::endl;
			float theta = length (v);
			v = normalize (v);

			float hc_quat[4];
			float w_hc_quat = 1.f;
			hc_quat[0] = std::sin (0.5*theta)*v[0];
			hc_quat[1] = std::sin (0.5*theta)*v[1];
			hc_quat[2] = std::sin (0.5*theta)*v[2];
			hc_quat[3] = std::cos (0.5*theta);


			e_x = cos (theta)*e_x + sin (theta)*cross (v, e_x) 
				+ (1.f - cos (theta))*dot (v, e_x)*v;
			e_y = cos (theta)*e_y + sin (theta)*cross (v, e_y) 
				+ (1.f - cos (theta))*dot (v, e_y)*v;
			e_z = cos (theta)*e_z + sin (theta)*cross (v, e_z) 
				+ (1.f - cos (theta))*dot (v, e_z)*v;
			//			for (int m = 0; m < 3; m++) {
			//				wc_r[m][0] = e_x[m]; wc_r[m][1] = e_y[m]; wc_r[m][2] = e_z[m];
			//			}

			//			float wc_rhs_val = wc_r[l][k];			
			//			float wc_rhs_val = e_z[k];
			float wc_rhs_val;
			//			if (k == 0)
			//				wc_rhs_val = e_z[0]/(1 - e_z[2]);
			//			else if (k == 1)
			//				wc_rhs_val = e_z[1]/(1 - e_z[2]);

			wc_rhs_val = w_hc_quat*hc_quat[k];		
			std::cout << "wc_rhs_val : " << wc_rhs_val << std::endl;


			cudaMemcpy (sys_rhs + wc_id, &wc_rhs_val, sizeof (float), 
									cudaMemcpyHostToDevice);
			CheckCUDAError ();
		}

		if (!update_only) {
			if (cu_csr_schol_info_ != NULL)
				cusolverSpDestroyCsrcholInfo (cu_csr_schol_info_);

			cusolverSpCreateCsrcholInfo (&cu_csr_schol_info_);

			cusolver_status = cusolverSpXcsrcholAnalysis (cusolver_handle_, sys_n, 
																										sys_nnz,
																										descr_bh_, 
																										sys_csr_row_inds, 
																										sys_csr_col_inds,
																										cu_csr_schol_info_);
			if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
				std::cout << "CUSolver Cholesky Analysis Failed" << std::endl;
			}

			size_t internal_data_in_bytes;
			size_t workspace_in_bytes;
			cusolver_status = cusolverSpScsrcholBufferInfo (cusolver_handle_, 
																											sys_n, sys_nnz,
																											descr_bh_, 
																											sys_csr_vals, 
																											sys_csr_row_inds, 
																											sys_csr_col_inds,
																											cu_csr_schol_info_, 
																											&internal_data_in_bytes, 
																											&workspace_in_bytes);
			if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
				std::cout << "CUSolver Cholesky Buffer Info Failed" << std::endl;
			}

			if (cu_csr_schol_p_buffer_ != NULL)
				cudaFree (cu_csr_schol_p_buffer_);
			cudaMalloc (&cu_csr_schol_p_buffer_, workspace_in_bytes);
			CheckCUDAError ();

			cusolver_status = cusolverSpScsrcholFactor (cusolver_handle_, 
																									sys_n, sys_nnz,
																									descr_bh_, 
																									sys_csr_vals, 
																									sys_csr_row_inds, 
																									sys_csr_col_inds,
																									cu_csr_schol_info_, 
																									cu_csr_schol_p_buffer_);
			if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
				std::cout << "CUSolver Cholesky Factorization Failed" << std::endl;
			}

			int position;
			cusolver_status = cusolverSpScsrcholZeroPivot (cusolver_handle_, 
																										 cu_csr_schol_info_, 
																										 tol, &position);
			if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
				std::cout << "CUSolver Cholesky Zero Pivot Failed" << std::endl;
			}
		}

		cudaEventRecord (start);
		cusolver_status = cusolverSpScsrcholSolve (cusolver_handle_, sys_n, sys_rhs,
																							 sys_x, cu_csr_schol_info_, 
																							 cu_csr_schol_p_buffer_);
		cudaEventRecord (stop);
		if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
			std::cout << "CUSolver Cholesky Solve Failed" << std::endl;
		}

		cudaEventSynchronize (stop);
		float milliseconds = 0;
		cudaEventElapsedTime (&milliseconds, start, stop);
		//		std::cout << "CUSolver Solve in " << milliseconds << "ms." << std::endl;

		cudaMemcpy (&host_x[0], sys_x, sys_n*sizeof (float), cudaMemcpyDeviceToHost);

		//		StrideGPUCopy (quats_pong_, sys_x, k, 4, sys_n);

		for (int i = 0; i < sys_n; i++) {
			q_field[4*i + k] = host_x[i];
		}
		//			cudaFree (p_buffer);
		//			cusolverSpDestroyCsrcholInfo (info);
	}

	cudaMemcpy (quats_pong_, &(q_field[0]), sys_n*sizeof (float4), cudaMemcpyHostToDevice);

	for (int i = 0; i < sys_n; i++) {
		float4 quat = make_float4 (q_field[4*i], q_field[4*i + 1], q_field[4*i + 2], q_field[4*i + 3]);
		quat = normalize (quat);
		Vec3f quat_xyz (quat.x, quat.y, quat.z);
		float quat_w = quat.w;
		float theta = 2.0*std::atan2 (quat_xyz.length (), quat_w);
		twist_field_ping_[i] = theta*normalize (quat_xyz);
	}
}

__global__ void  InitQuats (float * quats, const float4 & init_quat, int size) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= size)
		return;

	quats[4*id] = 0.f;
	quats[4*id + 1] = 1.f;
	quats[4*id + 2] = 0.f;
	quats[4*id + 3] = 0.f;
}

void FrameField::InitializeQuaternions (float * quats, const float4 & init_quat) {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	int size = dim_x*dim_y*dim_z;
	int block_dim = 512;
	int grid_dim = (size/block_dim) + 1;
	InitQuats<<<grid_dim, block_dim>>> 
		(quats, init_quat, size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
}


void FrameField::DebugLTLMatrix () {
	//	/*
	//	 * Debug L^T*L Matrix
	//	 */
	//	std::vector<float> x_host (4*sys_n, 0);
	//	std::vector<float> y_host (4*sys_n, 0);
	//	std::vector<float> lambda_host (sys_n, 0);
	//	std::vector<float> b_host (4*sys_n, 0);
	//	std::vector<float> ltl_host (sys_n*sys_n);
	//	std::vector<float> l_host (sys_n*sys_n);
	//	std::vector<float> residual_host (sys_n, 0);
	//	Eigen::MatrixXf mat_ltl_host (sys_n, sys_n);
	//	Eigen::MatrixXf mat_l_host (sys_n, sys_n);
	//	Eigen::MatrixXf mat_u_host (sys_n, sys_n);
	//	Eigen::MatrixXf diag_lambda_host (sys_n, sys_n);
	//	Eigen::MatrixXf diag_inv_identity_lambda_host (sys_n, sys_n);
	//	Eigen::MatrixXf diag_identity_host (sys_n, sys_n);
	//	float omega = 1.f/4.f;
	//	//	float omega = 1.f;
	//	//	omega /= 2.f;
	//	//	omega = (1.f/5.f);
	//
	//	for (int k = 0; k < 4; k++) {
	//		// Set (CPU) Right Hand Side vector
	//		// Weighted Constraints
	//		for (int c = 0; c < (int)hc_points_.size (); c++) {
	//			Vec3f wc_pos = hc_points_[c].position ();
	//			Vec3i wc_ipos (wc_pos[0], wc_pos[1], wc_pos[2]);
	//			Vec3f wc_twist = hc_points_[c].twist ();
	//			int wc_id = wc_ipos[0] + dim_x*wc_ipos[1] + dim_x*dim_y*wc_ipos[2];
	//			Vec3f v = wc_twist;
	//			float theta = length (v);
	//			v = normalize (v);
	//
	//			float hc_quat[4];
	//			float w_hc_quat = 1.f;
	//			hc_quat[0] = std::sin (0.5*theta)*v[0];
	//			hc_quat[1] = std::sin (0.5*theta)*v[1];
	//			hc_quat[2] = std::sin (0.5*theta)*v[2];
	//			hc_quat[3] = std::cos (0.5*theta);
	//
	//
	//			float wc_rhs_val;
	//			wc_rhs_val = hc_quat[k];		
	//			lambda_host[wc_id] = 1.f;
	//			b_host[wc_id + k*sys_n] = wc_rhs_val;
	//			std::cout << "wc_rhs_val : " << wc_rhs_val << std::endl;
	//		}
	//	}
	//
	//	float * dense_ltl = NULL;
	//	cudaMalloc (&dense_ltl, sys_n*sys_n*sizeof (float));
	//	CheckCUDAError ();
	//	cusparseStatus_t cusparse_status;
	//	cusparse_status = cusparseScsr2dense (cusparse_handle_, 
	//																				sys_n, sys_n, 
	//																				descr_bh_, csrv_ltl_, csrr_ltl_, csrc_ltl_, 
	//																				dense_ltl, 
	//																				sys_n);
	//	cudaDeviceSynchronize ();
	//	CheckCUDAError ();
	//	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
	//		std::cout << "CUSparse Operation failed." << std::endl;
	//	} else {
	//		std::cout << "CUSparse Operation done." << std::endl;
	//	}
	//
	//	cudaMemcpy (&(ltl_host[0]), dense_ltl, sys_n*sys_n*sizeof (float), 
	//							cudaMemcpyDeviceToHost);
	//	CheckCUDAError ();
	//
	//	float * dense_l = NULL;
	//	cudaMalloc (&dense_l, sys_n*sys_n*sizeof (float));
	//	CheckCUDAError ();
	//	cusparse_status = cusparseScsr2dense (cusparse_handle_, 
	//																				sys_n, sys_n, 
	//																				descr_bh_, csrv_l_, csrr_l_, csrc_l_, 
	//																				dense_l, 
	//																				sys_n);
	//	cudaDeviceSynchronize ();
	//	CheckCUDAError ();
	//	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
	//		std::cout << "CUSparse Operation failed." << std::endl;
	//	} else {
	//		std::cout << "CUSparse Operation done." << std::endl;
	//	}
	//
	//	cudaMemcpy (&(l_host[0]), dense_l, sys_n*sys_n*sizeof (float), 
	//							cudaMemcpyDeviceToHost);
	//	CheckCUDAError ();
	//
	//
	//	Eigen::VectorXf X (sys_n);
	//	Eigen::VectorXf Y (sys_n);
	//	Eigen::VectorXf Z (sys_n);
	//	Eigen::VectorXf U (sys_n);
	//	Eigen::VectorXf B (sys_n);
	//
	//	for (int i = 0; i < sys_n; i++)
	//		for (int j = 0; j < sys_n; j++) {
	//			mat_ltl_host (i, j) = ltl_host[i + j*sys_n];
	//			mat_l_host (i, j) = l_host[i + j*sys_n];
	//		}
	//
	//	mat_ltl_host = mat_l_host*mat_l_host;
	//
	//	for (int i = 0; i < sys_n; i++)
	//		for (int j = 0; j < sys_n; j++) {
	//			if (j != i) {
	//				diag_lambda_host (i, j) = 0.f;
	//				diag_identity_host (i, j) = 0.f;
	//				diag_inv_identity_lambda_host (i, j) = 0.f;
	//			} else {
	//				diag_lambda_host (i, j) = lambda_host[i];
	//				diag_identity_host (i, j) = 1.f;
	//				diag_inv_identity_lambda_host (i, j) = 1.f/(1.f + lambda_host[i]);
	//			}
	//		}
	//
	//	//	mat_u_host = diag_identity_host - omega*diag_inv_identity_lambda_host*(mat_ltl_host + diag_lambda_host);
	//	//	mat_u_host = diag_inv_identity_lambda_host*(mat_ltl_host + diag_lambda_host);
	//	//	mat_u_host = (1.f/(1.f + (1.f/6.f)))*mat_ltl_host - diag_identity_host;
	//	mat_u_host = mat_ltl_host + diag_lambda_host;
	//
	//	for (int i = 0; i < dim_x; i++)
	//		for (int j = 0; j < dim_y; j++)
	//			for (int k = 0; k < dim_z; k++) {
	//				if ((i == 8) && (j == 9) && (k == 4)) {
	//					//					x_host[i + dim_x*j + dim_x*dim_y*k] = 1.f;
	//				}
	//				if ((i == 0) && (j == 0) && (k == 0)) {
	//					//					x_host[i + dim_x*j + dim_x*dim_y*k] = 1.f;
	//				}
	//			}
	//
	//	//	int max_iter = 100000000;
	//	int max_iter = 0;
	//	for (int k = 0; k < 4; k++) {
	//		std::cout << k << "-k component optimization" << std::endl;
	//		bool has_converged = false;
	//		for (int i = 0; i < sys_n; i++) {
	//			//			X (i) = 0.f;
	//			X (i) = x_host[k*sys_n + i];
	//			//			X (i) = q_field[4*i + k];
	//			B (i) = b_host[k*sys_n + i];
	//		}
	//
	//		U = diag_lambda_host*B;
	//
	//		for (int iter = 0; (iter < max_iter) && !has_converged; iter++) {
	//			//			Y = X + omega*(U - mat_u_host*X);
	//			//			Y = mat_l_host.transpose ()*mat_l_host*X;
	//
	//			if (iter%40 == 0) {
	//				Z = mat_ltl_host*X + diag_lambda_host*X - diag_lambda_host*B;
	//				float max_error = Z.lpNorm<Eigen::Infinity> ();
	//				if (max_error < 0.00001)
	//					has_converged = true;
	//				std::cout << "iter : " << iter << " : " << Z.lpNorm<Eigen::Infinity> () << std::endl;
	//			}
	//			X = Y;
	//		}
	//
	//
	//		for (int i = 0; i < sys_n; i++) {
	//			y_host[k*sys_n + i] = X (i);
	//		}
	//	}
	//
	//	//	max_iter = 100000000;
	//	float mask_center_val = 1.f + (1.f/6.f);
	//	int m_range = 2;
	//	int dim_mask = 5;
	//
	//	std::cout << "bh_mask" << std::endl;
	//	int bh_curs = 0;
	//	for (int k = -m_range; k <= m_range; k++) {
	//		for (int j = -m_range; j <= m_range; j++) {
	//			for (int i = -m_range; i <= m_range; i++) {
	//				std::cout << bh_mask_vals_[bh_curs] << " ";	
	//				bh_curs++;
	//			}
	//			std::cout << std::endl;
	//		}
	//		std::cout << std::endl;
	//	}
	//
	//	Eigen::MatrixXf mat_5x5_l_host (dim_mask*dim_mask*dim_mask, 
	//																	dim_mask*dim_mask*dim_mask);
	//	Eigen::MatrixXf mat_5x5_ltl_host (dim_mask*dim_mask*dim_mask, 
	//																		dim_mask*dim_mask*dim_mask);
	//
	//	std::vector<Vec3i> mask_ids;
	//	std::vector<float> mask_vals;
	//
	//	mask_ids.clear ();
	//	mask_vals.clear ();
	//	mask_ids.push_back (Vec3i (0, 0, -1));
	//	mask_ids.push_back (Vec3i (0, -1, 0));
	//	mask_ids.push_back (Vec3i (-1, 0, 0));
	//	mask_ids.push_back (Vec3i (0, 0, 0));
	//	mask_ids.push_back (Vec3i (1, 0, 0));
	//	mask_ids.push_back (Vec3i (0, 1, 0));
	//	mask_ids.push_back (Vec3i (0, 0, 1));
	//	mask_vals.push_back (1.f);
	//	mask_vals.push_back (1.f);
	//	mask_vals.push_back (1.f);
	//	mask_vals.push_back (0.f);
	//	mask_vals.push_back (1.f);
	//	mask_vals.push_back (1.f);
	//	mask_vals.push_back (1.f);
	//
	//	mat_5x5_l_host = Eigen::MatrixXf::Zero (dim_mask*dim_mask*dim_mask, 
	//																					dim_mask*dim_mask*dim_mask);
	//	mat_5x5_ltl_host = Eigen::MatrixXf::Zero (dim_mask*dim_mask*dim_mask, 
	//																						dim_mask*dim_mask*dim_mask);
	//	for (int k = 0; k < dim_mask; k++)
	//		for (int j = 0; j < dim_mask; j++) 
	//			for (int i = 0; i < dim_mask; i++) {
	//				int cell_id = dim_mask*dim_mask*k + dim_mask*j + i;
	//				float sum_mask = 0.f;
	//				for (int l = 0; l < (int)mask_vals.size (); l++) {
	//					int im = i + mask_ids[l][0];
	//					int jm = j + mask_ids[l][1];
	//					int km = k + mask_ids[l][2];
	//					if ((0 <= im) && (im < dim_mask) &&
	//							(0 <= jm) && (jm < dim_mask) &&
	//							(0 <= km) && (km < dim_mask)) {
	//						sum_mask += mask_vals[l];
	//					}
	//				}
	//
	//				for (int l = 0; l < (int)mask_vals.size (); l++) {
	//					int im = i + mask_ids[l][0];
	//					int jm = j + mask_ids[l][1];
	//					int km = k + mask_ids[l][2];
	//					int mask_cell_id = dim_mask*dim_mask*km + dim_mask*jm + im;
	//					if ((0 <= im) && (im < dim_mask) &&
	//							(0 <= jm) && (jm < dim_mask) &&
	//							(0 <= km) && (km < dim_mask)) {
	//						float mask_val;
	//						if (cell_id != mask_cell_id)
	//							mask_val = -(mask_vals[l]/sum_mask);
	//						else {
	//							mask_val = 1.f - (mask_vals[l]/sum_mask);
	//						}
	//						mat_5x5_l_host (cell_id, mask_cell_id) = mask_val;
	//					}
	//				}
	//			}
	//
	//
	//	mat_5x5_ltl_host = mat_5x5_l_host.transpose ()*mat_5x5_l_host;
	//	//	mat_5x5_ltl_host = mat_5x5_l_host*mat_5x5_l_host;
	//
	//	//	std::cout << "mat_5x5_l_host" << std::endl;
	//	//	std::cout << mat_5x5_l_host << std::endl;
	//	//	std::cout << "mat_5x5_ltl_host" << std::endl;
	//	//	std::cout << mat_5x5_ltl_host << std::endl;
	//
	//	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver_5x5 (mat_5x5_ltl_host + Eigen::MatrixXf::Identity (5*5*5, 5*5*5));
	//
	//	if (eigensolver_5x5.info () != Eigen::Success) {
	//		std::cout << "impossible to compute eigen decomposition" << std::endl;
	//	} else {
	//		std::cout << "spectral radius : " << eigensolver_5x5.eigenvalues ().lpNorm<Eigen::Infinity> () << std::endl;
	//		std::cout << "max eigen value : " << eigensolver_5x5.eigenvalues ().maxCoeff () << std::endl;
	//		std::cout << "min eigen value : " << eigensolver_5x5.eigenvalues ().minCoeff () << std::endl;
	//	}
	//
	//
	//	//	std::cout << "L_infinity(mat_5x5_l_host, mat_l_host) : " 
	//	//		<< (mat_5x5_l_host - mat_l_host).lpNorm<Eigen::Infinity> () << std::endl;
	//	//	std::cout << "L_infinity(mat_5x5_ltl_host, mat_ltl_host) : " 
	//	//		<< (mat_5x5_ltl_host - mat_ltl_host).lpNorm<Eigen::Infinity> () << std::endl;	
	//	std::cout << "L_infinity(mat_l_host.transpose ()*mat_l_host, mat_ltl_host) : " 
	//		<< (mat_l_host.transpose ()*mat_l_host - mat_ltl_host).lpNorm<Eigen::Infinity> () << std::endl;
	//	std::cout << "L_infinity(mat_l_host*mat_l_host, mat_ltl_host) : " 
	//		<< (mat_l_host*mat_l_host - mat_ltl_host).lpNorm<Eigen::Infinity> () << std::endl;
	//
	//	//	std::cout << "mat_ltl_host (0, 0) : " << mat_ltl_host (0, 0) << std::endl;
	//	//	std::cout << "mat_5x5_ltl_host (0, 0) : " << mat_5x5_ltl_host (0, 0) << std::endl;
	//	//	
	//	//	std::cout << "mat_l_host (0, 0) : " << mat_l_host (0, 0) << std::endl;
	//	//	std::cout << "mat_5x5_l_host (0, 0) : " << mat_5x5_l_host (0, 0) << std::endl;
	//	//	
	//	//	std::cout << "mat_l_host (0, 1) : " << mat_l_host (0, 1) << std::endl;
	//	//	std::cout << "mat_5x5_l_host (0, 1) : " << mat_5x5_l_host (0, 1) << std::endl;
	//	//	
	//	//	std::cout << "mat_l_host (0, 5) : " << mat_l_host (0, 5) << std::endl;
	//	//	std::cout << "mat_5x5_l_host (0, 5) : " << mat_5x5_l_host (0, 5) << std::endl;
	//	//	
	//	//	std::cout << "mat_l_host (0, 25) : " << mat_l_host (0, 25) << std::endl;
	//	//	std::cout << "mat_5x5_l_host (0, 25) : " << mat_5x5_l_host (0, 25) << std::endl;
	//
	//
	//	std::vector< std::vector<float> > bh_masks;
	//	bh_masks.resize (dim_mask*dim_mask*dim_mask);
	//	for (int k = 0; k < dim_mask; k++)
	//		for (int j = 0; j < dim_mask; j++) 
	//			for (int i = 0; i < dim_mask; i++) {
	//				//				std::cout << "mask (" << i << ", " << j << ", " << k << ")" << std::endl;
	//				int cell_id = dim_mask*dim_mask*k + dim_mask*j + i;
	//				for (int k_n = -2; k_n <=2; k_n++) {
	//					for (int j_n = -2; j_n <=2; j_n++) {
	//						for (int i_n = -2; i_n <=2; i_n++) {
	//							int im = i + i_n;
	//							int jm = j + j_n;
	//							int km = k + k_n;
	//							int mask_cell_id = dim_mask*dim_mask*km + dim_mask*jm + im;
	//							if ((0 <= im) && (im < dim_mask) &&
	//									(0 <= jm) && (jm < dim_mask) &&
	//									(0 <= km) && (km < dim_mask)) {
	//								//								std::cout << mat_5x5_ltl_host (cell_id, mask_cell_id) << " ";
	//								bh_masks[cell_id].push_back (mat_5x5_ltl_host (cell_id, mask_cell_id));
	//							} else {
	//								bh_masks[cell_id].push_back (0);
	//								//								std::cout << 0 << " ";
	//							}
	//						}
	//						//						std::cout << std::endl;
	//					}
	//					//					std::cout << std::endl;
	//				}
	//				//				std::cout << std::endl;
	//			}
	//
	//	//	for (int k = 0; k < dim_mask; k++)
	//	//		for (int j = 0; j < dim_mask; j++) 
	//	//			for (int i = 0; i < dim_mask; i++) {
	//	//				int cell_id = dim_mask*dim_mask*k + dim_mask*j + i;
	//	//				std::cout << "mask (" << i << ", " << j << ", " << k << ") : of size " << bh_masks[cell_id].size () << std::endl;
	//	//				for (int k_n = -2; k_n <=2; k_n++) {
	//	//					for (int j_n = -2; j_n <=2; j_n++) {
	//	//						for (int i_n = -2; i_n <=2; i_n++) {
	//	//							int im = i + i_n;
	//	//							int jm = j + j_n;
	//	//							int km = k + k_n;
	//	//							int mask_cell_id = dim_mask*dim_mask*km + dim_mask*jm + im;
	//	//							int mask_id_neigh = dim_mask*dim_mask*(2 + k_n) + dim_mask*(2 + j_n) + (2 + i_n);
	//	//							if ((0 <= im) && (im < dim_mask) &&
	//	//									(0 <= jm) && (jm < dim_mask) &&
	//	//									(0 <= km) && (km < dim_mask) || true) {
	//	//								std::cout << bh_masks[cell_id][mask_id_neigh] << " "; 
	//	//							}
	//	//						}
	//	//						std::cout << std::endl;
	//	//					}
	//	//					std::cout << std::endl;
	//	//				}
	//	//				std::cout << std::endl;
	//	//			}
	//
	//
	//	omega = 1.f/6.f;
	//	//	max_iter = sys_n;
	//	//	max_iter = 100000000;
	//	max_iter = 1;
	//
	//	std::cout << "sys_n : " << sys_n << std::endl;
	//	Eigen::MatrixXf mat_ltl_by_mask_host (sys_n, sys_n);
	//
	//	for (int q = 0; q < 1; q++) {
	//		std::cout << q << "-k component optimization" << std::endl;
	//		bool has_converged = false;
	//		float * x_host_q = &(x_host[sys_n*q]);
	//		float * y_host_q = &(y_host[sys_n*q]);
	//		float * b_host_q = &(b_host[sys_n*q]);
	//
	//		for (int i = 0; i < sys_n; i++) {
	//			x_host_q[i] = 0.f;
	//			//			x_host_q[i] = 1.f;
	//			//			x_host_q[i] = q_field[4*i + q];
	//		}
	//
	//		//		for (int i = 0; i < sys_n; i++) {
	//		////			X (i) = 0.f;
	//		////			X (i) = x_host[q*sys_n + i];
	//		//			X (i) = q_field[4*i + q];
	//		//			B (i) = b_host[q*sys_n + i];
	//		//		}
	//		//
	//		//		U = diag_lambda_host*B;
	//
	//
	//		for (int iter = 0; (iter < max_iter) && !has_converged; iter++) {
	//			//			std::cout << "iter : " << iter << std::endl;
	//			//			for (int i = 0; i < sys_n; i++) {
	//			//				if (i == iter)	
	//			//					x_host_q[i] = 1.f;
	//			//				else
	//			//					x_host_q[i] = 0.f;
	//			//			}
	//
	//			//			Y = X + omega*(U - mat_u_host*X);
	//			//			X = Y;
	//
	//
	//			int3 res3 = make_int3 (dim_x, dim_y, dim_z);
	//			for (int i = 0; i < dim_x; i++) {
	//				for (int j = 0; j < dim_y; j++) {
	//#pragma omp parallel for
	//					for (int k = 0; k < dim_z; k++) {
	//						int3 id3 = make_int3 (i, j, k);
	//						int id = i + j*dim_x + k*dim_x*dim_y;
	//						float linear_operator_x = 0.f;
	//						float sum_mask_off_center = 0.f;
	//						//						bool debug = (i == 8) && (j == 9) && (k == 4);
	//						bool debug = (i == 0) && (j == 0) && (k == 0) && false;
	//
	//						if ((2 <= i) && (i < (dim_x - 2))
	//								&& (2 <= j) && (j < (dim_y - 2))
	//								&& (2 <= k) && (k < (dim_z - 2)) || true) {
	//
	//							int3 mask_id3;
	//							GridToMask (id3, res3, make_int3 (2, 2, 2), mask_id3);
	//							int mask_id = dim_mask*dim_mask*mask_id3.z + dim_mask*mask_id3.y + mask_id3.x;
	//
	//							if (debug) {
	//								std::cout << "mask id : " << mask_id3.x << ", " << mask_id3.y << ", " << mask_id3.z << std::endl;
	//							}
	//
	//							if (debug) {
	//								std::cout << "mask occupation : " << std::endl;
	//							}
	//							for (int k_n = -2; k_n <=2; k_n++) {
	//								for (int j_n = -2; j_n <=2; j_n++) {
	//									for (int i_n = -2; i_n <=2; i_n++) {
	//										int im = i + i_n;
	//										int jm = j + j_n;
	//										int km = k + k_n;
	//										int mask_id_neigh = dim_mask*dim_mask*(2 + k_n) + dim_mask*(2 + j_n) + (2 + i_n);
	//										float bh_mask_neigh_value = bh_masks[mask_id][mask_id_neigh];
	//										if (debug) {
	//											if (TestLimit (make_int3 (im, jm, km), res3))
	//												std::cout << 1 << " ";
	//											else
	//												std::cout << 0 << " ";
	//										}
	//									}
	//
	//									if (debug) {
	//										std::cout << std::endl;
	//									}
	//								}
	//								if (debug) {
	//									std::cout << std::endl;
	//								}
	//							}
	//
	//							if (debug) {
	//								std::cout << "mask values : " << std::endl;
	//							}
	//							for (int k_n = -2; k_n <=2; k_n++) {
	//								for (int j_n = -2; j_n <=2; j_n++) {
	//									for (int i_n = -2; i_n <=2; i_n++) {
	//										int im = i + i_n;
	//										int jm = j + j_n;
	//										int km = k + k_n;
	//										int id_neigh = dim_x*dim_y*km + dim_x*jm + im;
	//										int mask_id_neigh = dim_mask*dim_mask*(2 + k_n) + dim_mask*(2 + j_n) + (2 + i_n);
	//										float bh_mask_neigh_value = bh_masks[mask_id][mask_id_neigh];
	//										if (debug) {
	//											std::cout << bh_mask_neigh_value << " ";
	//										}
	//										if (TestLimit (make_int3 (im, jm, km), res3)) {
	//											linear_operator_x += bh_mask_neigh_value*x_host_q[id_neigh];
	//										}
	//									}
	//
	//									if (debug) {
	//										std::cout << std::endl;
	//									}
	//								}
	//								if (debug) {
	//									std::cout << std::endl;
	//								}
	//							}
	//						}
	//
	//
	//						linear_operator_x += lambda_host[id]*x_host_q[id];
	//						float residual = lambda_host[id]*b_host_q[id] - linear_operator_x;		
	//						y_host_q[id] = x_host_q[id] + omega*residual;
	//
	//						//						y_host_q[id] = linear_operator_x;
	//						//						residual_host[id] = linear_operator_x - mat_ltl_host (id, iter);
	//
	//						//						if (fabs (residual_host[id]) > 0.001f) {
	//						//							std::cout << "id : " << i << ", " << j << ", " << k << std::endl;
	//						//							std::cout << "residual : " << residual_host[id] << std::endl;
	//						//						}
	//						if (debug) {
	//							//							std::cout << "linear_operator_x : " << residual << std::endl;
	//							std::cout << "residual : " << residual_host[id] << std::endl;
	//						}
	//						residual_host[id] = residual;
	//					}
	//				}
	//			}
	//
	//			float max_residual = 0.f;
	//			for (int i = 0; i < sys_n; i++) {
	//				float residual = fabs (residual_host[i]);
	//				if (residual > max_residual)
	//					max_residual = fabs (residual);
	//			}
	//
	//			//			if ((max_residual < 0.00001f || max_residual > 10.f) && false)
	//			//				has_converged = true;
	//
	//			if (max_residual < 0.00000001f)
	//				has_converged = true;
	//
	//			if ((iter%100 == 0)) {
	//				std::cout << "max_residual : " << max_residual << std::endl;
	//			}
	//			for (int i = 0; i < sys_n; i++) 
	//				x_host_q[i] = y_host_q[i];
	//
	//
	//			float max_diff = 0.f;
	//			for (int i = 0; i < sys_n; i++) {
	//				float diff = fabs (x_host_q[i] - q_field[4*i]);
	//				if (diff > max_diff)
	//					max_diff = fabs (diff);
	//			}
	//
	//			if ((iter%100) == 0)
	//				std::cout << "max_diff : " << max_diff << std::endl;
	//
	//
	//			//			for (int i = 0; i < sys_n; i++) 
	//			//				mat_ltl_by_mask_host (i, iter) = y_host_q[i];
	//		}
	//	}
	//
	//	Eigen::MatrixXf mat_diff_host = mat_ltl_by_mask_host - mat_ltl_host;
	//
	//	std::cout << "L_infinity(mat_ltl_by_mask_host, mat_ltl_host) : " 
	//		<< mat_diff_host.lpNorm<Eigen::Infinity> () << std::endl;
	//	mat_diff_host = mat_ltl_by_mask_host - (mat_l_host.transpose ()*mat_l_host);
	//	std::cout << "L_infinity(mat_ltl_by_mask_host, mat_l_host.transpose ()*mat_l_host) : " 
	//		<< mat_diff_host.lpNorm<Eigen::Infinity> () << std::endl;
	//
	//
	//	for (int i = 0; i < dim_x; i++)
	//		for (int j = 0; j < dim_y; j++)
	//			for (int k = 0; k < dim_z; k++) {
	//				if ((i == 8) && (j == 9) && (k == 4)) {
	//					std::cout << "inside" << std::endl;
	//					for (int k_neigh = k - 4; k_neigh <= k + 4; k_neigh++) {
	//						for (int j_neigh = j - 4; j_neigh <= j + 4; j_neigh++) {
	//							for (int i_neigh = i - 4; i_neigh <= i + 4; i_neigh++) {
	//								int id_neigh = i_neigh + dim_x*j_neigh + dim_x*dim_y*k_neigh;
	//								float value = y_host[id_neigh];
	//								//								std::cout << "y_test[" << i_neigh << "][" << j_neigh << "][" << k_neigh << "] : " << 
	//								//									value << std::endl;
	//								std::cout << value << " ";
	//							}
	//							std::cout << std::endl;
	//						}
	//						std::cout << std::endl;
	//					}
	//				}
	//
	//				if ((i == 0) && (j == 0) && (k == 0)) {
	//					std::cout << "border" << std::endl;
	//					for (int k_neigh = k - 4; k_neigh <= k + 4; k_neigh++) {
	//						for (int j_neigh = j - 4; j_neigh <= j + 4; j_neigh++) {
	//							for (int i_neigh = i - 4; i_neigh <= i + 4; i_neigh++) {
	//								int id_neigh = i_neigh + dim_x*j_neigh + dim_x*dim_y*k_neigh;
	//								float value = 0.f;
	//								if ((0 <= i_neigh) && (i_neigh < dim_x)
	//										&& (0 <= j_neigh) && (j_neigh < dim_y)
	//										&& (0 <= k_neigh) && (k_neigh < dim_z)
	//									 )
	//									value = y_host[id_neigh];
	//								//								std::cout << "y_test[" << i_neigh << "][" << j_neigh << "][" << k_neigh << "] : " << 
	//								//									value << std::endl;
	//								std::cout << value << " ";
	//							}
	//							std::cout << std::endl;
	//						}
	//						std::cout << std::endl;
	//					}
	//				}
	//
	//			}
	//
	//	std::cout << "GPU Solution" << std::endl;
	//	for (int i = 0; i < dim_x; i++)
	//		for (int j = 0; j < dim_y; j++)
	//			for (int k = 0; k < dim_z; k++) {
	//				if ((i == 0) && (j == 0) && (k == 0)) {
	//					std::cout << "border" << std::endl;
	//					for (int k_neigh = k - 4; k_neigh <= k + 4; k_neigh++) {
	//						for (int j_neigh = j - 4; j_neigh <= j + 4; j_neigh++) {
	//							for (int i_neigh = i - 4; i_neigh <= i + 4; i_neigh++) {
	//								int id_neigh = i_neigh + dim_x*j_neigh + dim_x*dim_y*k_neigh;
	//								float value = 0.f;
	//
	//								if ((0 <= i_neigh) && (i_neigh < dim_x)
	//										&& (0 <= j_neigh) && (j_neigh < dim_y)
	//										&& (0 <= k_neigh) && (k_neigh < dim_z)
	//									 )
	//									value = q_field[4*id_neigh];
	//
	//								//								std::cout << "y_test[" << i_neigh << "][" << j_neigh << "][" << k_neigh << "] : " << 
	//								//									value << std::endl;
	//								std::cout << value << " ";
	//							}
	//							std::cout << std::endl;
	//						}
	//						std::cout << std::endl;
	//					}
	//				}
	//
	//				if ((i == 8) && (j == 9) && (k == 4)) {
	//					std::cout << "inside" << std::endl;
	//					for (int k_neigh = k - 4; k_neigh <= k + 4; k_neigh++) {
	//						for (int j_neigh = j - 4; j_neigh <= j + 4; j_neigh++) {
	//							for (int i_neigh = i - 4; i_neigh <= i + 4; i_neigh++) {
	//								int id_neigh = i_neigh + dim_x*j_neigh + dim_x*dim_y*k_neigh;
	//								float value = q_field[4*id_neigh];
	//								//								std::cout << "y_test[" << i_neigh << "][" << j_neigh << "][" << k_neigh << "] : " << 
	//								//									value << std::endl;
	//								std::cout << value << " ";
	//							}
	//							std::cout << std::endl;
	//						}
	//						std::cout << std::endl;
	//					}
	//				}
	//
	//			}
	//
	//	float max_diff = 0.f;
	//	for (int i = 0; i < sys_n; i++) {
	//		float diff = fabs (y_host[i] - q_field[4*i]);
	//		if (diff > max_diff)
	//			max_diff = fabs (diff);
	//	}
	//	std::cout << "max_diff : " << max_diff << std::endl;
	//
	//
	//	//	for (int i = 0; i < sys_n; i++) {
	//	//		float4 quat = make_float4 (y_host[0*sys_n + i], 
	//	//															 y_host[1*sys_n + i], 
	//	//															 y_host[2*sys_n + i], 
	//	//															 y_host[3*sys_n + i]);
	//	//		quat = normalize (quat);
	//	//		Vec3f quat_xyz (quat.x, quat.y, quat.z);
	//	//		float quat_w = quat.w;
	//	//		float theta = 2.0*std::atan2 (quat_xyz.length (), quat_w);
	//	//		twist_field_ping_[i] = theta*normalize (quat_xyz);
	//	//	}
	//
	//
	//	std::cout << "L^T*L Matrix Test" << std::endl;
	//
	//
}


