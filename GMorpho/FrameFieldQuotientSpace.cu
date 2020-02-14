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

#include <limits>
#include <fstream>

#include "CPoly/CPoly.h"

#include "FrameField.h"

using namespace MorphoGraphics;

Vec3f FrameField::QuatToTwist (float q0, float q1, float q2, float q3) {
	float4 quat = make_float4 (q0, q1, q2, q3);
	return QuatToTwist (quat);	
}

Vec3f FrameField::QuatToTwist (const float4 & quat) {
	Vec3f axis (quat.x, quat.y, quat.z);
	float axis_length = axis.length ();
	axis /= axis_length;
	float theta = 2.f*std::atan2 (axis_length, quat.w);	
	return theta*axis;
}

void FrameField::SO3ToQSO3 (const Vec3f & twist, Vec3f & qs2, 
														Vec3f & qs1) {
	Vec3f e_x, e_y, e_z;
	Vec3f e_x_symm, e_y_symm, e_z_symm;
	Vec3d nu, mu, rho, s2_a, s2_b;

	// (mu, rho, nu) is an orthonormal basis tangent of S^2
	// at nu.
	nu = Vec3d (0, 0, 1);
	mu = Vec3d (1, 0, 0); 
	rho = Vec3d (0, 1, 0); 

	// R = (e_x, e_y, e_z)
	TwistToRotation (twist, e_x, e_y, e_z);

	for (int i = 0; i < 1; i++) 
		for (int j = 0; j < 1; j++) 
			for (int k = 0; k < 1; k++) {
				RotationToSymmetricRotation (e_x, e_y, e_z, 
																		 i, j, k, 
																		 e_x_symm, 
																		 e_y_symm, 
																		 e_z_symm);


				// R := R^T
				Vec3d e_x_t, e_y_t, e_z_t;		
				e_x_t[0] = e_x_symm[0];
				e_x_t[1] = e_y_symm[0];
				e_x_t[2] = e_z_symm[0];
				
				e_y_t[0] = e_x_symm[1];
				e_y_t[1] = e_y_symm[1];
				e_y_t[2] = e_z_symm[1];

				e_z_t[0] = e_x_symm[2];
				e_z_t[1] = e_y_symm[2];
				e_z_t[2] = e_z_symm[2];

				std::cout << "e_x_t = " << e_x_t << std::endl;
				std::cout << "e_y_t = " << e_y_t << std::endl;
				std::cout << "e_z_t = " << e_z_t << std::endl;

				std::cout << "trace = " << e_x_t[0] + e_y_t[1] + e_z_t[2] << std::endl;

				// s2_a = R^T*mu
				// s2_b = R^T*rho
				for (int l = 0; l < 3; l++) {
					s2_a[l] = e_x_t[l]*mu[0] + 
						e_y_t[l]*mu[1] + 
						e_z_t[l]*mu[2];
					s2_b[l] = e_x_t[l]*rho[0] + 
						e_y_t[l]*rho[1] + 
						e_z_t[l]*rho[2];
				}
//				s2 = Vec3f (0, 0, 0);
//				for (int l = 0; l < 3; l++) {
//					s2[0] += e_x_symm[l]*nu[l];
//					s2[1] += e_y_symm[l]*nu[l];
//					s2[2] += e_z_symm[l]*nu[l];
//				}
				std::cout << "s2_a = " << s2_a << std::endl;
				std::cout << "s2_b = " << s2_b << std::endl;

				// c_a = P (s2_a);
				std::complex<double> c_a (s2_a[0]/(1.0 - s2_a[2]), 
																	s2_a[1]/(1.0 - s2_a[2]));
				std::complex<double> c_b (s2_b[0]/(1.0 - s2_b[2]), 
																	s2_b[1]/(1.0 - s2_b[2]));
				std::cout << "c_a = " << std::real (c_a) 
					<< ", " << std::imag (c_b) << std::endl;
				std::cout << "c_b = " << std::real (c_b) 
					<< ", " << std::imag (c_b) << std::endl;

				// w_a = F(c_a)
				// w_b = F(c_b)
				std::complex<double> w_a, w_b;
				w_a = std::pow (std::pow (c_a, 8) + 14.0*std::pow (c_a, 4) + 1.0, 3);
				w_a /= std::pow (108.0*c_a*(std::pow (c_a, 4) - 1.0), 4);
				
				w_b = std::pow (std::pow (c_b, 8) + 14.0*std::pow (c_b, 4) + 1.0, 3);
				w_b /= std::pow (108.0*c_b*(std::pow (c_b, 4) - 1.0), 4);

				std::cout << "w_a = " << std::real (w_a) 
					<< ", " << std::imag (w_a) << std::endl;
				std::cout << "w_b = " << std::real (w_b) 
					<< ", " << std::imag (w_b) << std::endl;

				// s2w_a = P^(-1)*F(c_a)
				// s2w_b = P^(-1)*F(c_b)
				double sqw_a = std::real (w_a)*std::real (w_a) + 
					std::imag (w_a)*std::imag (w_a);

				Vec3f s2w_a (2.0*std::real (w_a)/(sqw_a + 1.0), 
									 2.0*std::imag (w_a)/(sqw_a + 1.0), 
									 (sqw_a - 1.0)/(sqw_a + 1.0));
				std::cout << "s2w_a = " << s2w_a << std::endl;

				double sqw_b = std::real (w_b)*std::real (w_b) + 
					std::imag (w_b)*std::imag (w_b);

				Vec3f s2w_b (2.0*std::real (w_b)/(sqw_b + 1.0), 
									 2.0*std::imag (w_b)/(sqw_b + 1.0), 
									 (sqw_b - 1.0)/(sqw_b + 1.0));
				std::cout << "s2w_b = " << s2w_b << std::endl;
				std::cout << "dot (s2w_a, s2w_b) : " << dot (s2w_a, s2w_b) << std::endl;

//				w_b += std::complex<double> (0.001, -0.0012);

				qs2 = s2w_a;

				//				// Parallel Transport of tangent vectors mu and rho 
				//				// at nu to s2 = R*nu
				//				Vec3d para_twist, para_mu, para_rho;
				//				para_twist = std::acos (dot (nu, s2))*normalize (cross (nu, s2));
				//				para_mu = Rotate (para_twist, mu);
				//				para_rho = Rotate (para_twist, rho);
				//				std::cout << "para_mu = " << para_mu << std::endl;
				//				std::cout << "para_rho = " << para_rho << std::endl;
				//				std::cout << "dot (para_mu, s2) = " << dot (para_mu, s2) << std::endl;

				//				// Measure the remaining angle/direction in S^1
				//				// of R = [e_x_t; e_y_t; e_z_t] at this point.
				//				Vec3d gamma (dot (e_x_t, para_mu), 
				//										 dot (e_x_t, para_rho), 0);
				//				double para_theta = std::atan2 (gamma[1], gamma[0]);
				//				//				std::cout << "dot (s2, e_x_t) = " << dot (s2, e_x_t) << std::endl;
				//				//				std::cout << "dot (s2, e_y_t) = " << dot (s2, e_y_t) << std::endl;
				//				std::cout << "para_theta : " << para_theta << std::endl;
				//				std::cout << "gamma : " << gamma << std::endl;

				// Inverse Path.
				// a) Set up a the complex polynomial
				std::complex<double> wbis_a = w_a*(108.0*108.0*108.0*108.0);
				std::vector< std::complex<double> > coeffs (25, 0.0);
				std::complex<double> wbis_b = w_b*(108.0*108.0*108.0*108.0);
				double * opr = new double[25];
				double * opi = new double[25];
				double * zeror = new double[25];
				double * zeroi = new double[25];

				coeffs[24 - (24)] = 1.0;
				coeffs[24 - (20)] = (28.0 + 14.0) - wbis_a;
				coeffs[24 - (16)] = (2.0 + 14.0*14.0 + 14.0*28.0 + 1.0) - (-4.0*wbis_a);
				coeffs[24 - (12)] = (28.0 + 14.0*(2.0 + 14.0*14.0) + 28.0) - (6.0*wbis_a);
				coeffs[24 - (8)] = (1 + 14.0*28.0 + (2.0 + 14.0*14.0)) - (-4.0*wbis_a);
				coeffs[24 - (4)] = (14.0 + 28.0) - (wbis_a);
				coeffs[24 - (0)] = 1.0;

				for (int l = 0; l < 25; l++) {
					opr[l] = std::real (coeffs[l]);
					opi[l] = std::imag (coeffs[l]);
					//					std::cout << "coeff[" << i << "] : " 
					//						<< opr[i] << ", " << opi[i] << std::endl;
				}
				//				std::complex<double> rh, lh, grh, glh;
				//				lh = std::pow (c, 24) 
				//					+ (28.0 + 14.0)*std::pow (c, 20)
				//					+ (2.0 + 14.0*14.0 + 14.0*28.0 + 1.0)*std::pow (c, 16)
				//					+ (28.0 + 14.0*(2.0 + 14.0*14.0) + 28.0)*std::pow (c, 12)
				//					+ (1 + 14.0*28.0 + (2.0 + 14.0*14.0))*std::pow (c, 8)
				//					+ (14.0 + 28.0)*std::pow (c, 4)
				//					+ 1.0;
				//
				////				lh = (std::pow (c, 16) 
				////					+ 28.0*std::pow (c, 12)
				////					+ (2.0 + 14.0*14.0)*std::pow (c, 8)
				////					+ 28.0*std::pow (c, 4)
				////					+ 1.0);
				//
				//				rh = std::pow (c, 20)
				//					+ (-4.0)*std::pow (c, 16)
				//					+ 6.0*std::pow (c, 12)
				//					+ (-4.0)*std::pow (c, 8)
				//					+ std::pow (c, 4);
				//				rh *= w*(108.0*108.0*108.0*108.0);
				//
				//				glh = std::pow (std::pow (c, 8) + 14.0*std::pow (c, 4) + 1.0, 3);
				//				grh = w*std::pow (108.0*c*(std::pow (c, 4) - 1.0), 4);
				//
				//				std::cout << "lh = " << std::real (lh) 
				//					<< ", " << std::imag (lh) << std::endl;
				//				std::cout << "rh = " << std::real (rh) 
				//					<< ", " << std::imag (rh) << std::endl;
				//				std::cout << "glh = " << std::real (glh) 
				//					<< ", " << std::imag (glh) << std::endl;
				//				std::cout << "grh = " << std::real (grh) 
				//					<< ", " << std::imag (grh) << std::endl;

				std::vector<Vec3d> rs2_a, rs2_b;
				int num_root_found_a = cpoly (opr, opi, 24, zeror, zeroi);
				std::cout << "found_a " << num_root_found_a << " roots" << std::endl;
				for (int l = 0; l < num_root_found_a; l++) {
					std::complex<long double> root (zeror[l], zeroi[l]);
					//					std::cout << "zeros[" << i << "] : " 
					//						<< zeror[i] << ", " << zeroi[i] << std::endl;					
					// Get the direction in S^2 by inverse stereographic projection
					double sqr = std::real (root)*std::real (root) + 
						std::imag (root)*std::imag (root);
					Vec3d rs2 (2.0*std::real (root)/(sqr + 1.0), 
										 2.0*std::imag (root)/(sqr + 1.0), 
										 (sqr - 1.0)/(sqr + 1.0));
					rs2_a.push_back (rs2);
					//					std::cout << "rs2 = " << rs2 << std::endl;

					//					// Find the two remaining directions of the orthonormal
					//					// basis through parallel transport.
					//					// a) Parallel Transport of tangent vectors mu and rho 
					//					// at nu to roots2
					//					Vec3d d_mu (mu[0], mu[1], mu[2]);
					//					Vec3d d_rho (rho[0], rho[1], rho[2]);
					//					Vec3d d_nu (nu[0], nu[1], nu[2]);
					//					Vec3d rs2_para_twist, rs2_para_mu, rs2_para_rho;
					//					rs2_para_twist = std::acos (dot (d_nu, rs2))*normalize (cross (d_nu, rs2));
					//					rs2_para_mu = Rotate (rs2_para_twist, d_mu);
					//					rs2_para_rho = Rotate (rs2_para_twist, d_rho);
					//					//					std::cout << "rs2_para_mu = " << rs2_para_mu << std::endl;
					//					//					std::cout << "rs2_para_rho = " << rs2_para_rho << std::endl;
					//
					//					Vec3d mu_rs2 = std::cos (para_theta)*rs2_para_mu
					//						+ std::sin (para_theta)*rs2_para_rho;
					//					Vec3d rho_rs2 = -std::sin (para_theta)*rs2_para_mu
					//						+ std::cos (para_theta)*rs2_para_rho;
					//					//					std::cout << "mu_rs2 = " << mu_rs2 << std::endl;
					//					//					std::cout << "rho_rs2 = " << rho_rs2 << std::endl;
					//					//					std::cout << "nu_rs2 = " << rs2 << std::endl;
				}

				coeffs[24 - (24)] = 1.0;
				coeffs[24 - (20)] = (28.0 + 14.0) - wbis_b;
				coeffs[24 - (16)] = (2.0 + 14.0*14.0 + 14.0*28.0 + 1.0) - (-4.0*wbis_b);
				coeffs[24 - (12)] = (28.0 + 14.0*(2.0 + 14.0*14.0) + 28.0) - (6.0*wbis_b);
				coeffs[24 - (8)] = (1 + 14.0*28.0 + (2.0 + 14.0*14.0)) - (-4.0*wbis_b);
				coeffs[24 - (4)] = (14.0 + 28.0) - (wbis_b);
				coeffs[24 - (0)] = 1.0;

				for (int l = 0; l < 25; l++) {
					opr[l] = std::real (coeffs[l]);
					opi[l] = std::imag (coeffs[l]);
					//					std::cout << "coeff[" << i << "] : " 
					//						<< opr[i] << ", " << opi[i] << std::endl;
				}

				int num_root_found_b = cpoly (opr, opi, 24, zeror, zeroi);
				std::cout << "found_b " << num_root_found_b << " roots" << std::endl;
				for (int l = 0; l < num_root_found_b; l++) {
					std::complex<long double> root (zeror[l], zeroi[l]);
					//					std::cout << "zeros[" << i << "] : " 
					//						<< zeror[i] << ", " << zeroi[i] << std::endl;					
					// Get the direction in S^2 by inverse stereographic projection
					double sqr = std::real (root)*std::real (root) + 
						std::imag (root)*std::imag (root);
					Vec3d rs2 (2.0*std::real (root)/(sqr + 1.0), 
										 2.0*std::imag (root)/(sqr + 1.0), 
										 (sqr - 1.0)/(sqr + 1.0));
					rs2_b.push_back (rs2);
					//					std::cout << "rs2 = " << rs2 << std::endl;
				}

				int l_abs_dot = 0, m_abs_dot = 0;
				double min_abs_dot = std::numeric_limits<double>::max ();
				for (int l = 0; l < (int)rs2_a.size (); l++)
					for (int m = 0; m < (int)rs2_b.size (); m++) {
						double abs_dot = std::abs (dot (rs2_a[l], rs2_b[m]));
						std::cout << "dot (" << rs2_a[l] << "; " << rs2_b[m] << ") : " 
							<< dot (rs2_a[l], rs2_b[m]) << std::endl;
						if (abs_dot < min_abs_dot) {
							min_abs_dot = abs_dot;
							l_abs_dot = l;
							m_abs_dot = m;
						}
					}

				int count_valid = 0;
				for (int l = 0; l < (int)rs2_a.size (); l++)
					for (int m = 0; m < (int)rs2_b.size (); m++) {
						double abs_dot = std::abs (dot (rs2_a[l], rs2_b[m]));
						std::cout << "dot (" << rs2_a[l] << "; " << rs2_b[m] << ") : " 
							<< dot (rs2_a[l], rs2_b[m]) << std::endl;
						if (std::abs (abs_dot - min_abs_dot) < abs_dot/10.0) {
							count_valid++;
						}
					}

				std::cout << "rs2_a[l_abs_dot] : " << rs2_a[l_abs_dot] << std::endl;
				std::cout << "rs2_b[m_abs_dot] : " << rs2_b[m_abs_dot] << std::endl;
				std::cout << "count_valid : " << count_valid << std::endl;
				std::cout << "min_abs_dot : " << min_abs_dot << std::endl;

				delete opr;
				delete opi;	
				delete zeror;
				delete zeroi;	

				std::cout << "e_x_t = " << e_x_t << std::endl;
				std::cout << "e_y_t = " << e_y_t << std::endl;
				std::cout << "e_z_t = " << e_z_t << std::endl;
			}
}


void FrameField::InitExploreSO3 (const Vec3f & bbox_min, const Vec3f & bbox_max, 
																 const Vec3i & res) {
	bbox_min_ = bbox_min;
	bbox_max_ = bbox_max;
	res_ = res;
	cell_size_ = (bbox_max_[0] - bbox_min_[0])/res_[0];

	slice_xy_id_ = 0;
	max_hc_number_ = 20;
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
	lambda_ = 10.f;

	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	int num_cells = dim_x*dim_y*dim_z;
	int num_hcs = hc_points_.size ();

	twist_field_ping_.resize (num_cells);
	position_field_ping_.resize (num_cells);
	twist_field_pong_.resize (num_cells);
	position_field_pong_.resize (num_cells);

	float dx = (bbox_max_[0] - bbox_min_[0])/dim_x;
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
			}

	num_tris_in_cube_ = 12;	
	opengl_frame_field_p_.reserve (3*num_cells*num_tris_in_cube_);
	opengl_frame_field_n_.reserve (3*num_cells*num_tris_in_cube_);
	ComputeOpenGLCube ();
}

void FrameField::ExploreSO3 (int k_x, int k_y, int k_z) {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	float twist_min = -M_PI;
	float twist_max = M_PI;
	float pos_dx = (bbox_max_[0] - bbox_min_[0])/dim_x;
	float pos_scale = pos_dx;
	float twist_dx = (twist_max - twist_min)/dim_x;
	float twist_scale = twist_dx;

#pragma omp parallel for
	for (int i = 0; i < dim_x; i++)
		for (int j = 0; j < dim_y; j++)
			for (int k = 0; k < dim_z; k++) {
				int cell_id = i + dim_x*j + dim_x*dim_y*k;
				Vec3f twist = twist_scale*Vec3f (i, j, k) 
					+ Vec3f (twist_min, twist_min, twist_min) 
					+ 0.5f*Vec3f (twist_dx, twist_dx, twist_dx);
				Vec3f pos = pos_scale*Vec3f (i, j, k) 
					+ bbox_min_
					+ 0.5f*Vec3f (pos_dx, pos_dx, pos_dx);
				//				twist[0] = 2*M_PI*(((float) rand ())/RAND_MAX - 0.5f);
				//				twist[1] = 2*M_PI*(((float) rand ())/RAND_MAX - 0.5f);
				//				twist[2] = 2*M_PI*(((float) rand ())/RAND_MAX - 0.5f);
				Vec3f factored_twist = TwistToFactoredTwist (twist);
				Vec3f symmetric_twist;
				if ((cell_id%2 == 0) || true)
					symmetric_twist = factored_twist;
				else
					symmetric_twist = TwistToSymmetricTwist (factored_twist, k_x, k_y, k_z);

				//				Vec3f e_x, e_y, e_z;
				//				TwistToRotation (twist, e_x, e_y, e_z);
				//
				//				Vec3f cayley_twist = RotationToTwistCayley (e_x, e_y, e_z);
				//				Vec3f eig_twist = RotationToTwistEigenDecomp (e_x, e_y, e_z);

				Vec3f final_twist = symmetric_twist;
				Vec3f initial_twist = factored_twist;

				twist_field_ping_[cell_id] = final_twist;
				twist_field_pong_[cell_id] = initial_twist;
				if (length (twist) < M_PI 
						//						&& (length (factored_twist) >= M_PI/4.f)
					 ) {
					position_field_ping_[cell_id] = (1.f/twist_max)*final_twist;
					position_field_pong_[cell_id] = (1.f/twist_max)*initial_twist;
				} else {
					position_field_ping_[cell_id] = Vec3f (0.f, 0.f, 0.f);
					position_field_pong_[cell_id] = Vec3f (0.f, 0.f, 0.f);
				}
				//				position_field_ping_[cell_id] = pos;
				//		position_field_ping_[i] = twist 
				//			+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]);
				//				position_field_ping_[i] = twist;
			}
	bool output_to_file = true;
	std::vector<Vec3i> symm_ids;
	int symm_idx[24] = {0, 1, 3, 4, 5, 7, 12, 13, 15, 16, 17, 19, 25, 26, 27, 
		2, 6, 8, 9, 10, 11, 14, 18, 24};

	int curs_symm_idx = 0;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			for (int k = 0; k < 4; k++) {
				bool found = false;
				for (int idx = 0; idx < 2; idx++) {
					if (symm_idx[idx] == curs_symm_idx)
						found = true;
				}
				if (found)
					symm_ids.push_back (Vec3i (i, j, k));
				curs_symm_idx++;
			}


	if (output_to_file) {
		for (int symm = 0; symm < symm_ids.size (); symm++) {
			std::ostringstream filename;
			filename << "frames_color_quat_symm_" << symm << ".ply";
			std::ofstream ply_file;
			ply_file.open (filename.str ().c_str ());
			int num_cells = dim_x*dim_y*dim_z;
			float dx = (bbox_max_[0] - bbox_min_[0])/dim_x;
			float scale = 0.5f*dx;
			//			float alpha = 1.f;
			unsigned int ver_count = num_cells*cube_p_.size ();
			unsigned int tri_count = (num_cells*cube_p_.size ())/3;
			ply_file << "ply" << std::endl 
				<< "format ascii 1.0" << std::endl 
				<< "element vertex " << ver_count << std::endl
				<< "property float x" << std::endl
				<< "property float y" << std::endl
				<< "property float z" << std::endl
				<< "property uchar red" << std::endl
				<< "property uchar green" << std::endl
				<< "property uchar blue" << std::endl
				//		<< "property float nx" << std::endl
				//		<< "property float ny" << std::endl
				//		<< "property float nz" << std::endl
				<< "element face " << tri_count << std::endl
				<< "property list uchar int vertex_index" << std::endl
				<< "end_header" << std::endl;

			//	for (int i = 0; i < num_cells; i++) {
			//		Vec3f p = position_field_ping_[i];
			//		Vec3f twist = twist_field_ping_[i];
			//		twist = 255.f*0.5f*(Vec3f (1.f, 1.f, 1.f) + (1.f/twist_max)*twist);
			//		Vec3i color = Vec3i (twist[0], twist[1], twist[2]);
			//		//		Vec3Df p = mesh.V ()[i].p;
			//		//		Vec3Df n = mesh.V ()[i].n;
			//
			//		ply_file
			//			<< p[0] << " " << p[1] << " " << p[2] << " "
			//			<< color[0] << " " << color[1] << " " << color[2] << " "
			//			//			<< n[0] << " " << n[1] << " " << n[2] << " " 
			//			<< std::endl; 
			//	}

			//	int k_x = 2, k_y = 0, k_y = 0;
			for (int i = 0; i < dim_x; i++)
				for (int j = 0; j < dim_y; j++) 
					for (int k = 0; k < dim_z; k++) {
						for (int l = 0; l < (int)cube_p_.size (); l++) {
							Vec3f pos, normal, v, pos_transf, normal_transf, colorf;
							//					Vec3f pos_transf_x_pi;
							Vec3i color;
							// Pass 0 : adjust scale
							pos = scale*cube_p_[l];
							normal = cube_n_[l];	
							// Pass 1 :adjust orientation
							Vec3f twist = twist_field_ping_[i + dim_x*j + dim_x*dim_y*k];
							Vec3f symmetric_twist = TwistToSymmetricTwist (twist, 
																														 symm_ids[symm][0], 
																														 symm_ids[symm][1], 
																														 symm_ids[symm][2]);
							v = symmetric_twist;
							//							v = twist;
							colorf = twist;
							colorf = 255.f*0.5f*(Vec3f (1.f, 1.f, 1.f) + 2.f*(1.f/twist_max)*colorf);
							color = Vec3i (colorf[0], colorf[1], colorf[2]);
							float theta = length (v);
							v = normalize (v);
							pos_transf = cos (theta)*pos + sin (theta)*cross (v, pos)
								+ (1.f - cos (theta))*dot (v, pos)*v;
							normal_transf = cos (theta)*normal + sin (theta)*cross (v, normal)
								+ (1.f - cos (theta))*dot (v, normal)*v;

							toColor (2.f*(1.f - (float)cos (0.5f*theta)), colorf);
							color = Vec3i (255.f*colorf[0], 255.f*colorf[1], 255.f*colorf[2]);

							// Pass 1 bis : X axis rotation of Pi
							//					v = ((float)M_PI)*Vec3f (1, 0, 0);
							//					theta = length (v);
							//					v = normalize (v);
							//					pos_transf = cos (theta)*pos_transf 
							//						+ sin (theta)*cross (v, pos_transf)
							//						+ (1.f - cos (theta))*dot (v, pos_transf)*v;

							// Pass 2 :adjust position
							//							Vec3f translation = (1.f/twist_max)*symmetric_twist;
							Vec3f translation = sin (0.5f*theta)*v;
							//							Vec3f translation = (1.f/twist_max)*twist;
							//					Vec3f translation = position_field_ping_[i + dim_x*j + dim_x*dim_y*k];
							//						translation[0] = -translation[0];
							//					translation[1] = -translation[1];
							//					translation[2] = -translation[2];

							// Pass 2 bis : Z axis rotation of Pi/2
							//						v = 0.25f*((float)M_PI)*Vec3f (1, 0, 0);
							//						theta = length (v);
							//						v = normalize (v);
							//						translation = cos (theta)*translation 
							//							+ sin (theta)*cross (v, translation)
							//							+ (1.f - cos (theta))*dot (v, translation)*v;
							//
							pos_transf = pos_transf 
								+ translation;

							ply_file << pos_transf[0] << " " << pos_transf[1] << " " << pos_transf[2] << " "
								<< color[0] << " " << color[1] << " " << color[2] << " "
								<< std::endl;
						}
					}

			for (unsigned int i = 0; i < tri_count; i++) {
				ply_file << 3 << " " << 3*i << " "  << 3*i + 1 << " " << 3*i + 2 << std::endl;
			}
			ply_file.close ();

		}
	}
}

void FrameField::ComputeOpenGLFrameFieldPingPongInterpol (float alpha) {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	float dx = (bbox_max_[0] - bbox_min_[0])/dim_x;
	float scale = 0.5f*dx;

	opengl_frame_field_p_.clear ();
	opengl_frame_field_n_.clear ();

	//	for (int i = 0; i < dim_x; i++)
	//		for (int j = 0; j < dim_y; j++)
	//			for (int k = 0; k < dim_z; k++) {
	//				Vec3f pos = scale*Vec3f (i, j, k) 
	//					+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]) 
	//					+ 0.5f*Vec3f (dx, dx, dx);
	//				position_field_ping_[i + dim_x*j + dim_x*dim_y*k] = pos;
	//			}

	for (int i = 0; i < dim_x; i++)
		for (int j = 0; j < dim_y; j++) 
			for (int k = 0; k < dim_z; k++) {
				for (int l = 0; l < (int)cube_p_.size (); l++) {
					Vec3f pos, normal, v, pos_transf, normal_transf;
					// Pass 0 : adjust scale
					pos = scale*cube_p_[l];
					normal = cube_n_[l];	
					// Pass 1 :adjust orientation
					v = twist_field_ping_[i + dim_x*j + dim_x*dim_y*k];
					float theta = length (v);
					v = normalize (v);
					pos_transf = cos (theta)*pos + sin (theta)*cross (v, pos)
						+ (1.f - cos (theta))*dot (v, pos)*v;
					normal_transf = cos (theta)*normal + sin (theta)*cross (v, normal)
						+ (1.f - cos (theta))*dot (v, normal)*v;
					// Pass 2 :adjust position
					pos_transf = pos_transf 
						+ alpha*position_field_ping_[i + dim_x*j + dim_x*dim_y*k]
						+ (1.f - alpha)*position_field_pong_[i + dim_x*j + dim_x*dim_y*k];
						//					pos_transf = pos_transf + scale*Vec3f (i, j, k)
						//						+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]) + 0.5f*Vec3f (dx, dx, dx);
						opengl_frame_field_p_.push_back (pos_transf);
						opengl_frame_field_n_.push_back (normal_transf);
				}
			}
}


