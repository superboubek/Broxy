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

#include <cmath>
#include <cfloat>
#include <omp.h>

#include <cuda_profiler_api.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include "timing.h"
#include "cuda_math.h"

#include "FrameField.h"

using namespace MorphoGraphics;

void FrameField::print (const std::string & msg) {
	std::cout << "[FrameField] : " << msg << std::endl;
}

void FrameField::CheckCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if(err != cudaSuccess) {
		FrameField::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw FrameField::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}

/*
 * Accessors and Mutators
 */

void FrameField::AddHardConstrainedPoint (const Vec3f & position, 
																					const Vec3f & e_x, 
																					const Vec3f & e_y, 
																					const Vec3f & e_z) {
	Vec3f twist = RotationToTwistCayley (e_x, e_y, e_z);
	AddHardConstrainedPoint (position, twist);
}

void FrameField::AddHardConstrainedPoint (const Vec3f & position, 
																					const Vec3f & twist) {
	if (hc_points_.size () < max_hc_number_) {
		Vec3f grid_position = position;
		grid_position -= bbox_min_;
		grid_position = (1.f/cell_size_)*grid_position;
		for (int i = 0; i < 3; i++)
			grid_position[i] = std::floor (grid_position[i]);

		std::cout << "add point at : " << position << " | " << grid_position << std::endl;
		FramePoint hc_point (grid_position, twist);
		hc_points_.push_back (hc_point);
	}
}

void FrameField::UpdateHardConstrainedPoint (int i, 
																						 const Vec3f & e_x, 
																						 const Vec3f & e_y, 
																						 const Vec3f & e_z) {
	Vec3f twist = RotationToTwistCayley (e_x, e_y, e_z);
	UpdateHardConstrainedPoint (i, twist);
}

void FrameField::UpdateHardConstrainedPoint (int i, const Vec3f & twist) {
	if ((0 <= i) && (i < (int) hc_points_.size ())) {
		FramePoint hc_point (hc_points_[i].position (), twist);
		hc_points_[i] = hc_point;
	}
}

Vec3f FrameField::GetHardConstrainedPointTwist (int i) {
	if ((0 <= i) && (i < (int) hc_points_.size ()))
		return hc_points_[i].twist ();
	else
		return Vec3f (0, 0, 0);
}

int FrameField::GetHardConstrainedPointId (const Vec3f & position) {
	return GetHardConstrainedPointId (position, cell_size_);
}

int FrameField::GetHardConstrainedPointId (const Vec3f & position, float eps_tolerance) {
	int id_min = -1;
	float min_dist = std::numeric_limits<float>::max ();

	for (int i = 0; i < (int) hc_points_.size (); i++) {
		Vec3f hc_grid_position = hc_points_[i].position ();
		Vec3f hc_position = cell_size_*hc_grid_position;
		hc_position += bbox_min_;
		float dist = length (position - hc_position);
		//		std::cout << "query point at : " << position << std::endl;
		//		std::cout << "check point at : " << hc_position << " | " << hc_grid_position << std::endl;
		//		std::cout << "dist : " << dist << " eps : " << eps_tolerance << std::endl;
		if (dist < eps_tolerance)
			if (dist < min_dist) {
				min_dist = dist;
				id_min = i;	
			}
	}

	return id_min;
}


/*
 * Utility functions related to SO3 and other symmetry
 * spaces
 */

Vec3f FrameField::RotationToTwistStable (const Vec3f & e_x, 
																				 const Vec3f & e_y, 
																				 const Vec3f & e_z) {
	float r00, r01, r02, r10, r11, r12, r20, r21, r22;
	r00 = e_x[0]; r01 = e_y[0]; r02 = e_z[0];
	r10 = e_x[1]; r11 = e_y[1]; r12 = e_z[1];
	r20 = e_x[2]; r21 = e_y[2]; r22 = e_z[2];

	Vec3f axis;
	if (r00 >= r11)
		if (r00 >= r22)
			axis = Vec3f (r00 + 1, (r01 + r10)/2, (r02 + r20)/2);
		else
			axis = Vec3f ((r20 + r02)/2,  (r21 + r12)/2,  r22 + 1);
	else
		if (r11 >= r22)
			axis = Vec3f ((r10 + r01)/2, r11 + 1, (r12 + r21)/2);
		else
			axis = Vec3f ((r20 + r02)/2, (r21 + r12)/2, r22 + 1);
	axis = normalize (axis);

	Vec3f axis_plain (r21 - r12, r02 - r20, r10 - r01);
	float cos_theta = std::min (std::max ((r00 + r11 + r22 - 1.0)/2.0, -1.0), 1.0);
	float theta = copysign (std::acos (cos_theta), axis_plain[0]*axis[0]);

	std::cout << "Robust axis : " << axis << std::endl;
	std::cout << "Robust angle : " << theta << std::endl;
	std::cout << "Robust angle*axis : " << theta*axis << std::endl;

	return theta*axis;
}

Vec3f FrameField::RotationToTwistEigenDecomp (const Vec3f & e_x, 
																							const Vec3f & e_y, 
																							const Vec3f & e_z) {
	Eigen::Matrix3f rot;
	rot (0, 0) = e_x[0]; rot (0, 1) = e_y[0]; rot (0, 2) = e_z[0];
	rot (1, 0) = e_x[1]; rot (1, 1) = e_y[1]; rot (1, 2) = e_z[1];
	rot (2, 0) = e_x[2]; rot (2, 1) = e_y[2]; rot (2, 2) = e_z[2];

	Eigen::EigenSolver<Eigen::Matrix3f> es (rot);
	//	std::cout << "R is:" << std::endl << rot << std::endl;
	//	std::cout << "The eigenvalues of R are:" << std::endl << es.eigenvalues() << std::endl;
	//	std::cout << "The matrix of eigenvectors, V, is:" << std::endl << es.eigenvectors() << std::endl << std::endl;
	Eigen::Vector3cf v_complex;
	for (int k = 0; k < 3; k++) {
		if (std::abs (es.eigenvalues ()[k].imag ()) < 1e-10)
			v_complex = es.eigenvectors ().col (k);
	}
	//	float cos_theta_decomp = es.eigenvalues ()[1].real ();
	//	float sin_theta_decomp = es.eigenvalues ()[1].imag ();
	Vec3f axis (std::real (v_complex (0)), 
							std::real (v_complex (1)), 
							std::real (v_complex (2)));
	//	float theta_decomp = std::atan2 (sin_theta_decomp, cos_theta_decomp);
	//	std::cout << "cos, sin : " << cos (theta_x) << " | " << sin (theta_x) << std::endl;
	//	std::cout << "Decomp cos, sin : " << cos_theta_decomp << " | " << sin_theta_decomp << std::endl;

	Vec3f axis_plain (rot (2, 1) - rot (1, 2), 
										rot (0, 2) - rot (2, 0), 
										rot (1, 0) - rot (0, 1));
	float cos_theta = (rot (0, 0) + rot (1, 1) + rot (2, 2) - 1.0)/2.0;
	cos_theta = std::min (std::max (cos_theta, -1.f), 1.f);
	float theta = copysign (std::acos (cos_theta), 
													axis_plain[0]*axis[0]);

	return theta*axis;
}

Vec3d FrameField::RotationToTwistCayley (const Vec3d & e_x, 
																				 const Vec3d & e_y, 
																				 const Vec3d & e_z) {
	Eigen::Matrix3d rot;
	rot (0, 0) = e_x[0]; rot (0, 1) = e_y[0]; rot (0, 2) = e_z[0];
	rot (1, 0) = e_x[1]; rot (1, 1) = e_y[1]; rot (1, 2) = e_z[1];
	rot (2, 0) = e_x[2]; rot (2, 1) = e_y[2]; rot (2, 2) = e_z[2];

	Eigen::Matrix3d cayley;
	cayley = (rot - Eigen::MatrixXd::Identity (3, 3))
		*(rot + Eigen::MatrixXd::Identity (3, 3)).inverse ();			
	Vec3d axis (-cayley (1, 2), cayley (0, 2), -cayley (0, 1));

	double cos_theta = (rot (0, 0) + rot (1, 1) + rot (2, 2) - 1.0)/2.0;
	double theta = std::acos (std::min (std::max (cos_theta, -1.0), 1.0));

	axis = normalize (axis);
	Vec3d final_axis (theta*axis[0], theta*axis[1], theta*axis[2]);

	//	std::cout << "Cayley Transform : " << std::endl << cayley << std::endl;
	//	std::cout << "Cayley axis : " << axis << std::endl;
	//	std::cout << "Cayley angle : " << theta << std::endl;

	return final_axis;
}

Vec3f FrameField::RotationToTwistCayley (const Vec3f & e_x, 
																				 const Vec3f & e_y, 
																				 const Vec3f & e_z) {
	Eigen::Matrix3d rot;
	rot (0, 0) = e_x[0]; rot (0, 1) = e_y[0]; rot (0, 2) = e_z[0];
	rot (1, 0) = e_x[1]; rot (1, 1) = e_y[1]; rot (1, 2) = e_z[1];
	rot (2, 0) = e_x[2]; rot (2, 1) = e_y[2]; rot (2, 2) = e_z[2];

	Eigen::Matrix3d cayley;
	cayley = (rot - Eigen::MatrixXd::Identity (3, 3))
		*(rot + Eigen::MatrixXd::Identity (3, 3)).inverse ();			
	Vec3d axis (-cayley (1, 2), cayley (0, 2), -cayley (0, 1));

	double cos_theta = (rot (0, 0) + rot (1, 1) + rot (2, 2) - 1.0)/2.0;
	double theta = std::acos (std::min (std::max (cos_theta, -1.0), 1.0));

	axis = normalize (axis);
	Vec3f final_axis (theta*axis[0], theta*axis[1], theta*axis[2]);

	//	std::cout << "Cayley Transform : " << std::endl << cayley << std::endl;
	//	std::cout << "Cayley axis : " << axis << std::endl;
	//	std::cout << "Cayley angle : " << theta << std::endl;

	return final_axis;
}

Vec3f FrameField::TwistToFactoredTwist (const Vec3f & twist) {
	Vec3d v (twist[0], twist[1], twist[2]);
	double theta = length (v);
	v = normalize (v);

	Vec3d e_x (1, 0, 0);
	Vec3d e_y (0, 1, 0);
	Vec3d e_z (0, 0, 1);

	e_x = cos (theta)*e_x + sin (theta)*cross (v, e_x) 
		+ (1.f - cos (theta))*dot (v, e_x)*v;
	e_y = cos (theta)*e_y + sin (theta)*cross (v, e_y) 
		+ (1.f - cos (theta))*dot (v, e_y)*v;
	e_z = cos (theta)*e_z + sin (theta)*cross (v, e_z) 
		+ (1.f - cos (theta))*dot (v, e_z)*v;

	return RotationToFactoredTwist (e_x, e_y, e_z);
}

Vec3f FrameField::RotationToFactoredTwist (const Vec3d & e_x, 
																					 const Vec3d & e_y, 
																					 const Vec3d & e_z) {
	int n_fold = 4;
	Vec3d min_twist_d = RotationToTwistCayley (e_x, e_y, e_z);
	Vec3f min_twist (min_twist_d[0], min_twist_d[1], min_twist_d[2]);
	float min_twist_mag = length (min_twist);

	double r[3][3];
	for (int i = 0; i < n_fold; i++)
		for (int j = 0; j < n_fold; j++)
			for (int k = 0; k < n_fold; k++) {
				double ci = cos (i*(2.f*M_PI/((double)n_fold)));
				double si = sin (i*(2.f*M_PI/((double)n_fold)));
				double cj = cos (j*(2.f*M_PI/((double)n_fold)));
				double sj = sin (j*(2.f*M_PI/((double)n_fold)));
				double ck = cos (k*(2.f*M_PI/((double)n_fold)));
				double sk = sin (k*(2.f*M_PI/((double)n_fold)));
				double r_x[3][3], r_y[3][3], r_z[3][3], r_ijk[3][3];

				for (int l = 0; l < 3; l++) {
					r[l][0] = e_x[l]; r[l][1] = e_y[l]; r[l][2] = e_z[l];
				}

				r_x[0][0] = 1.f;	r_x[0][1] = 0.f;	r_x[0][2] = 0.f;
				r_x[1][0] = 0.f;	r_x[1][1] = ci; 	r_x[1][2] = -si;
				r_x[2][0] = 0.f;	r_x[2][1] = si; 	r_x[2][2] = ci;

				r_y[0][0] = cj;		r_y[0][1] = 0.f;	r_y[0][2] = sj;
				r_y[1][0] = 0.f;	r_y[1][1] = 1.f; 	r_y[1][2] = 0.f;
				r_y[2][0] = -sj;	r_y[2][1] = 0.f; 	r_y[2][2] = cj;

				r_z[0][0] = ck;		r_z[0][1] = -sk;	r_z[0][2] = 0.f;
				r_z[1][0] = sk;		r_z[1][1] = ck; 	r_z[1][2] = 0.f;
				r_z[2][0] = 0.f;	r_z[2][1] = 0.f; 	r_z[2][2] = 1.f;

				for (int l = 0; l < 3; l++) 
					for (int m = 0; m < 3; m++) {
						double val = 0.f;
						for (int s = 0; s < 3; s++) 
							val += r[l][s]*r_x[s][m];
						r_ijk[l][m]	= val;
					}
				for (int l = 0; l < 3; l++) 
					for (int m = 0; m < 3; m++)
						r[l][m]	= r_ijk[l][m];

				for (int l = 0; l < 3; l++) 
					for (int m = 0; m < 3; m++) {
						double val = 0.f;
						for (int s = 0; s < 3; s++) 
							val += r[l][s]*r_y[s][m];
						r_ijk[l][m]	= val;
					}
				for (int l = 0; l < 3; l++) 
					for (int m = 0; m < 3; m++)
						r[l][m]	= r_ijk[l][m];

				for (int l = 0; l < 3; l++) 
					for (int m = 0; m < 3; m++) {
						double val = 0.f;
						for (int s = 0; s < 3; s++) 
							val += r[l][s]*r_z[s][m];
						r_ijk[l][m]	= val;
					}

				Vec3d e_x_ijk, e_y_ijk, e_z_ijk;

				for (int l = 0; l < 3; l++) {
					e_x_ijk[l] = r_ijk[l][0]; e_y_ijk[l] = r_ijk[l][1]; e_z_ijk[l] = r_ijk[l][2];
				}

				Vec3d twist_ijk_d = RotationToTwistCayley (e_x_ijk, e_y_ijk, e_z_ijk);
				Vec3f twist_ijk (twist_ijk_d[0], twist_ijk_d[1], twist_ijk_d[2]);
				float twist_ijk_mag = length (twist_ijk);

				if ((twist_ijk_mag < min_twist_mag)
					 ) {
					min_twist_mag = twist_ijk_mag;
					min_twist = twist_ijk;
				}
			}
	return min_twist;
}

Vec3f FrameField::RotationToFactoredTwist (const Vec3f & e_x, 
																					 const Vec3f & e_y, 
																					 const Vec3f & e_z) {
	Vec3d e_x_d (e_x[0], e_x[1], e_x[2]);
	Vec3d e_y_d (e_y[0], e_y[1], e_y[2]);
	Vec3d e_z_d (e_z[0], e_z[1], e_z[2]);
	return RotationToFactoredTwist (e_x_d, e_y_d, e_z_d);
}

void FrameField::TwistToRotation (const Vec3f & twist, 
																	Vec3f & e_x, Vec3f & e_y, Vec3f & e_z) {
	Vec3f v = twist;
	float theta = length (v);
	v = normalize (v);

	e_x = Vec3f (1, 0, 0);
	e_y = Vec3f (0, 1, 0);
	e_z = Vec3f (0, 0, 1);

	e_x = cos (theta)*e_x + sin (theta)*cross (v, e_x) 
		+ (1.f - cos (theta))*dot (v, e_x)*v;
	e_y = cos (theta)*e_y + sin (theta)*cross (v, e_y) 
		+ (1.f - cos (theta))*dot (v, e_y)*v;
	e_z = cos (theta)*e_z + sin (theta)*cross (v, e_z) 
		+ (1.f - cos (theta))*dot (v, e_z)*v;
}

Vec3f FrameField::TwistToSymmetricTwist (const Vec3f & twist, 
																				 int k_x, int k_y, int k_z) {
	int i = k_x, j = k_y, k = k_z;
	float ci = cos (i*(M_PI/2.f));
	float si = sin (i*(M_PI/2.f));
	float cj = cos (j*(M_PI/2.f));
	float sj = sin (j*(M_PI/2.f));
	float ck = cos (k*(M_PI/2.f));
	float sk = sin (k*(M_PI/2.f));
	float r[3][3], r_x[3][3], r_y[3][3], r_z[3][3], r_ijk[3][3];

	Vec3f e_x, e_y, e_z;
	TwistToRotation (twist, e_x, e_y, e_z);

	for (int l = 0; l < 3; l++) {
		r[l][0] = e_x[l]; r[l][1] = e_y[l]; r[l][2] = e_z[l];
	}

	r_x[0][0] = 1.f;	r_x[0][1] = 0.f;	r_x[0][2] = 0.f;
	r_x[1][0] = 0.f;	r_x[1][1] = ci; 	r_x[1][2] = -si;
	r_x[2][0] = 0.f;	r_x[2][1] = si; 	r_x[2][2] = ci;

	r_y[0][0] = cj;		r_y[0][1] = 0.f;	r_y[0][2] = sj;
	r_y[1][0] = 0.f;	r_y[1][1] = 1.f; 	r_y[1][2] = 0.f;
	r_y[2][0] = -sj;	r_y[2][1] = 0.f; 	r_y[2][2] = cj;

	r_z[0][0] = ck;		r_z[0][1] = -sk;	r_z[0][2] = 0.f;
	r_z[1][0] = sk;		r_z[1][1] = ck; 	r_z[1][2] = 0.f;
	r_z[2][0] = 0.f;	r_z[2][1] = 0.f; 	r_z[2][2] = 1.f;

	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++) {
			float val = 0.f;
			for (int s = 0; s < 3; s++) 
				val += r[l][s]*r_x[s][m];
			r_ijk[l][m]	= val;
		}
	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++)
			r[l][m]	= r_ijk[l][m];

	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++) {
			float val = 0.f;
			for (int s = 0; s < 3; s++) 
				val += r[l][s]*r_y[s][m];
			r_ijk[l][m]	= val;
		}
	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++)
			r[l][m]	= r_ijk[l][m];

	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++) {
			float val = 0.f;
			for (int s = 0; s < 3; s++) 
				val += r[l][s]*r_z[s][m];
			r_ijk[l][m]	= val;
		}
	//				for (int l = 0; l < 3; l++) 
	//					for (int m = 0; m < 3; m++)
	//						r[l][m]	= r_ijk[l][m];

	Vec3f e_x_ijk, e_y_ijk, e_z_ijk;

	for (int l = 0; l < 3; l++) {
		e_x_ijk[l] = r_ijk[l][0]; e_y_ijk[l] = r_ijk[l][1]; e_z_ijk[l] = r_ijk[l][2];
	}

	Vec3f twist_ijk = RotationToTwistCayley (e_x_ijk, e_y_ijk, e_z_ijk);
	return twist_ijk;
}

void FrameField::RotationToSymmetricRotation (const Vec3f & e_x, 
																							const Vec3f & e_y, 
																							const Vec3f & e_z, 
																							int k_x, int k_y, int k_z, 
																							Vec3f & e_x_symm, 
																							Vec3f & e_y_symm, 
																							Vec3f & e_z_symm) {
	int i = k_x, j = k_y, k = k_z;
	float ci = cos (i*(M_PI/2.f));
	float si = sin (i*(M_PI/2.f));
	float cj = cos (j*(M_PI/2.f));
	float sj = sin (j*(M_PI/2.f));
	float ck = cos (k*(M_PI/2.f));
	float sk = sin (k*(M_PI/2.f));
	float r[3][3], r_x[3][3], r_y[3][3], r_z[3][3], r_ijk[3][3];

	for (int l = 0; l < 3; l++) {
		r[l][0] = e_x[l]; r[l][1] = e_y[l]; r[l][2] = e_z[l];
	}

	r_x[0][0] = 1.f;	r_x[0][1] = 0.f;	r_x[0][2] = 0.f;
	r_x[1][0] = 0.f;	r_x[1][1] = ci; 	r_x[1][2] = -si;
	r_x[2][0] = 0.f;	r_x[2][1] = si; 	r_x[2][2] = ci;

	r_y[0][0] = cj;		r_y[0][1] = 0.f;	r_y[0][2] = sj;
	r_y[1][0] = 0.f;	r_y[1][1] = 1.f; 	r_y[1][2] = 0.f;
	r_y[2][0] = -sj;	r_y[2][1] = 0.f; 	r_y[2][2] = cj;

	r_z[0][0] = ck;		r_z[0][1] = -sk;	r_z[0][2] = 0.f;
	r_z[1][0] = sk;		r_z[1][1] = ck; 	r_z[1][2] = 0.f;
	r_z[2][0] = 0.f;	r_z[2][1] = 0.f; 	r_z[2][2] = 1.f;

	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++) {
			float val = 0.f;
			for (int s = 0; s < 3; s++) 
				val += r[l][s]*r_x[s][m];
			r_ijk[l][m]	= val;
		}
	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++)
			r[l][m]	= r_ijk[l][m];

	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++) {
			float val = 0.f;
			for (int s = 0; s < 3; s++) 
				val += r[l][s]*r_y[s][m];
			r_ijk[l][m]	= val;
		}
	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++)
			r[l][m]	= r_ijk[l][m];

	for (int l = 0; l < 3; l++) 
		for (int m = 0; m < 3; m++) {
			float val = 0.f;
			for (int s = 0; s < 3; s++) 
				val += r[l][s]*r_z[s][m];
			r_ijk[l][m]	= val;
		}

	for (int l = 0; l < 3; l++) {
		e_x_symm[l] = r_ijk[l][0]; 
		e_y_symm[l] = r_ijk[l][1]; 
		e_z_symm[l] = r_ijk[l][2];
	}
}

Vec3f FrameField::Rotate (const Vec3f & twist, const Vec3f & v) {
	Vec3f u = twist, v_rot;
	float theta = length (u);
	u = normalize (u);
	v_rot = std::cos (theta)*v + std::sin (theta)*cross (u, v) 
		+ (1.f - std::cos (theta))*dot (u, v)*u;
	return v_rot;
}

Vec3d FrameField::Rotate (const Vec3d & twist, const Vec3d & v) {
	Vec3d u = twist, v_rot;
	double theta = length (u);
	u = normalize (u);
	v_rot = std::cos (theta)*v + std::sin (theta)*cross (u, v) 
		+ (1.0 - std::cos (theta))*dot (u, v)*u;
	return v_rot;
}

void FrameField::Init (const Vec3f & bbox_min, const Vec3f & bbox_max, 
											 const Vec3i & res, OptimMethod optim_method) {	
	optim_method_ = optim_method;

	if (optim_method_ == BiHarmonicSystem)	{
		InitSparseBiharmonicSystem (bbox_min, bbox_max, res);
	} else if (optim_method_ == LocalOptimization) {
		InitLocalOptimization (bbox_min, bbox_max, res);
	}
}

void FrameField::Generate (bool update_only, bool use_group_symmetry) {
	if (optim_method_ == BiHarmonicSystem) {
		if (!update_only) {
			BuildSparseBiharmonicSystem ();
		}

		if (hc_points_.size () != 0) {
			SolveSparseBiharmonicSystem (update_only);
		} else {
			float4 init_quat = make_float4 (0, 1, 0, 0);
			InitializeQuaternions (quats_pong_, init_quat);
		}
	} else if (optim_method_ == LocalOptimization) {
		//		SolveLocalOptimization (10000);
		SolveLocalOptimization (100, use_group_symmetry);
		//		SolveLocalOptimization (10000, use_group_symmetry);
		//		SolveLocalOptimization (100000, use_group_symmetry);
		//		SolveLocalOptimization (1000000, use_group_symmetry);
	}
}

void FrameField::ComputeOpenGLFrameField () {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	float dx = cell_size_;
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
					pos_transf = pos_transf + position_field_ping_[i + dim_x*j + dim_x*dim_y*k];
					//					pos_transf = pos_transf + scale*Vec3f (i, j, k)
					//						+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]) + 0.5f*Vec3f (dx, dx, dx);
					opengl_frame_field_p_.push_back (pos_transf);
					opengl_frame_field_n_.push_back (normal_transf);
				}
			}
}

void FrameField::ComputeOpenGLFrameFieldSliceXY () {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	float dx = cell_size_;
	float scale = dx;

	opengl_frame_field_p_.clear ();
	opengl_frame_field_n_.clear ();

	int k = slice_xy_id_;

	if (k >= dim_z)
		return;

	//	for (int i = 0; i < dim_x; i++)
	//		for (int j = 0; j < dim_y; j++) {
	//			Vec3f pos = scale*Vec3f (i, j, k) 
	//				+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]) 
	//				+ 0.5f*Vec3f (dx, dx, dx);
	//			position_field_ping_[i + dim_x*j + dim_x*dim_y*k] = pos;
	//		}

	for (int i = 0; i < dim_x; i++)
		for (int j = 0; j < dim_y; j++) 
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
				pos_transf = pos_transf + position_field_ping_[i + dim_x*j + dim_x*dim_y*k];
				//				pos_transf = pos_transf + scale*Vec3f (i, j, k)
				//					+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]) + 0.5f*Vec3f (dx, dx, dx);
				opengl_frame_field_p_.push_back (pos_transf);
				opengl_frame_field_n_.push_back (normal_transf);
			}
}

void FrameField::ComputeOpenGLHardConstrainedFrames () {
	int dim_x = res_[0]; int dim_y = res_[1]; int dim_z = res_[2];
	float dx = cell_size_;
	float scale = dx;

	opengl_hc_frames_p_.clear ();
	opengl_hc_frames_n_.clear ();

	for (int c = 0; c < (int) hc_points_.size (); c++)
		for (int l = 0; l < (int)cube_p_.size (); l++) {
			Vec3f pos, normal, v, pos_transf, normal_transf;
			// Pass 0 : adjust scale
			pos = scale*cube_p_[l];
			normal = cube_n_[l];	
			// Pass 1 :adjust orientation
			v = hc_points_[c].twist ();
			float theta = length (v);
			v = normalize (v);
			pos_transf = cos (theta)*pos + sin (theta)*cross (v, pos)
				+ (1.f - cos (theta))*dot (v, pos)*v;
			normal_transf = cos (theta)*normal + sin (theta)*cross (v, normal)
				+ (1.f - cos (theta))*dot (v, normal)*v;
			// Pass 2 :adjust position
			pos_transf = pos_transf + scale*hc_points_[c].position ()
				+ Vec3f (bbox_min_[0], bbox_min_[1], bbox_min_[2]) + 0.5f*Vec3f (dx, dx, dx);
			opengl_hc_frames_p_.push_back (pos_transf);
			opengl_hc_frames_n_.push_back (normal_transf);
		}
}

void FrameField::ComputeOpenGLCube () {
	std::vector<Vec3f> cube_p_list, cube_p, cube_n;

	//	int num_circ_points = 20;
	//	for (int i = 0; i < num_circ_points; i++) {
	//		cube_p_list.push_back (Vec3f (0.5f*cos (i*2.0*M_PI/(num_circ_points)), 
	//																	0.5f*sin (i*2.0*M_PI/(num_circ_points)), 
	//																	0.0));		
	//	}
	//	cube_p_list.push_back (Vec3f (0.0, 0.0, 1.0));
	//	cube_p_list.push_back (Vec3f (0.0, 0.0, 0.0));
	//
	//	for (int i = 0; i < num_circ_points; i++) {
	//		cube_p.push_back (cube_p_list[i]);
	//		cube_p.push_back (cube_p_list[(i + 1)%num_circ_points]);
	//		cube_p.push_back (cube_p_list[num_circ_points]);
	//	}
	//
	//	for (int i = 0; i < num_circ_points; i++) {
	//		cube_p.push_back (cube_p_list[i]);
	//		cube_p.push_back (cube_p_list[(i + 1)%num_circ_points]);
	//		cube_p.push_back (cube_p_list[num_circ_points + 1]);
	//	}

	//	cube_p_list.push_back (Vec3f (0.0, 0.5, 0.5));
	//	cube_p_list.push_back (Vec3f (0.0, 0.5, -0.5));
	//	cube_p_list.push_back (Vec3f (0.0, 0.0, -0.5));
	//	cube_p_list.push_back (Vec3f (1.0, 0.0, 0.0));
	//
	//	cube_p.push_back (cube_p_list[0]);
	//	cube_p.push_back (cube_p_list[1]);
	//	cube_p.push_back (cube_p_list[2]);
	//
	//	cube_p.push_back (cube_p_list[0]);
	//	cube_p.push_back (cube_p_list[1]);
	//	cube_p.push_back (cube_p_list[3]);
	//
	//	cube_p.push_back (cube_p_list[1]);
	//	cube_p.push_back (cube_p_list[2]);
	//	cube_p.push_back (cube_p_list[3]);
	//	
	//	cube_p.push_back (cube_p_list[0]);
	//	cube_p.push_back (cube_p_list[3]);
	//	cube_p.push_back (cube_p_list[2]);

	//	cube_p_list.push_back (Vec3f (0.5, 0.5, 0.5));
	//	cube_p_list.push_back (Vec3f (-0.5, 0.5, 0.5));
	//	cube_p_list.push_back (Vec3f (0.5, -0.5, 0.5));
	//	cube_p_list.push_back (Vec3f (-0.5, -0.5, 0.5));
	//	cube_p_list.push_back (Vec3f (0.5, 0.5, -0.5));
	//	cube_p_list.push_back (Vec3f (-0.5, 0.5, -0.5));
	//	cube_p_list.push_back (Vec3f (0.5, -0.5, -0.5));
	//	cube_p_list.push_back (Vec3f (-0.5, -0.5, -0.5));

	float scale_xy = 0.5f;
	cube_p_list.push_back (Vec3f (0.5*scale_xy, 0.5*scale_xy, 0.5));
	cube_p_list.push_back (Vec3f (-0.5*scale_xy, 0.5*scale_xy, 0.5));
	cube_p_list.push_back (Vec3f (0.5*scale_xy, -0.5*scale_xy, 0.5));
	cube_p_list.push_back (Vec3f (-0.5*scale_xy, -0.5*scale_xy, 0.5));
	cube_p_list.push_back (Vec3f (0.5*scale_xy, 0.5*scale_xy, -0.5));
	cube_p_list.push_back (Vec3f (-0.5*scale_xy, 0.5*scale_xy, -0.5));
	cube_p_list.push_back (Vec3f (0.5*scale_xy, -0.5*scale_xy, -0.5));
	cube_p_list.push_back (Vec3f (-0.5*scale_xy, -0.5*scale_xy, -0.5));

	cube_p.push_back (cube_p_list[0]);
	cube_p.push_back (cube_p_list[1]);
	cube_p.push_back (cube_p_list[2]);

	cube_p.push_back (cube_p_list[3]);
	cube_p.push_back (cube_p_list[2]);
	cube_p.push_back (cube_p_list[1]);

	cube_p.push_back (cube_p_list[0]);
	cube_p.push_back (cube_p_list[2]);
	cube_p.push_back (cube_p_list[4]);

	cube_p.push_back (cube_p_list[6]);
	cube_p.push_back (cube_p_list[4]);
	cube_p.push_back (cube_p_list[2]);

	cube_p.push_back (cube_p_list[0]);
	cube_p.push_back (cube_p_list[4]);
	cube_p.push_back (cube_p_list[1]);

	cube_p.push_back (cube_p_list[5]);
	cube_p.push_back (cube_p_list[1]);
	cube_p.push_back (cube_p_list[4]);

	cube_p.push_back (cube_p_list[7]);
	cube_p.push_back (cube_p_list[5]);
	cube_p.push_back (cube_p_list[6]);

	cube_p.push_back (cube_p_list[4]);
	cube_p.push_back (cube_p_list[6]);
	cube_p.push_back (cube_p_list[5]);

	cube_p.push_back (cube_p_list[7]);
	cube_p.push_back (cube_p_list[6]);
	cube_p.push_back (cube_p_list[3]);

	cube_p.push_back (cube_p_list[2]);
	cube_p.push_back (cube_p_list[3]);
	cube_p.push_back (cube_p_list[6]);

	cube_p.push_back (cube_p_list[7]);
	cube_p.push_back (cube_p_list[3]);
	cube_p.push_back (cube_p_list[5]);

	cube_p.push_back (cube_p_list[1]);
	cube_p.push_back (cube_p_list[5]);
	cube_p.push_back (cube_p_list[3]);

	for (int i = 0; i < (int) cube_p.size ()/3; i++) {
		Vec3f tn = normalize (cross (cube_p[3*i + 1] - cube_p[3*i], 
																 cube_p[3*i + 2] - cube_p[3*i]));
		cube_n.push_back (tn);
		cube_n.push_back (tn);
		cube_n.push_back (tn);
	}
	cube_p_ = cube_p;
	cube_n_ = cube_n;
}


