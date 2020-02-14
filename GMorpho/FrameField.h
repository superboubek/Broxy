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

#ifndef  FrameField_INC
#define  FrameField_INC

#include <vector>
#include <complex>
#include <Common/Vec3.h>

#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include "cuda_math.h"

namespace MorphoGraphics {
class FrameField {
	class Exception {
	public:
		inline Exception (const std::string & msg) : msg_ ("FrameField Error: " + msg) {}
		inline const std::string & msg () const { return msg_; }
	protected:
		std::string msg_;
	};

	class FramePoint {
	public:
		FramePoint () {
			position_ = Vec3f (0, 0, 0);
			twist_ = Vec3f (0, 0, 0);
		}
		FramePoint (const Vec3f & position, const Vec3f & twist) {
			position_ = position;
			twist_ = twist;
		}
		~FramePoint () {

		}
		inline const Vec3f & position () const { return position_; }
		inline const Vec3f & twist () const { return twist_; }
		inline void set_position (const Vec3f & position) { position_ = position; }
		inline void set_twist (const Vec3f & twist) { twist_ = twist; }
	private:
		Vec3f position_;
		Vec3f twist_;
	};

public:
	typedef enum {BiHarmonicSystem, LocalOptimization} OptimMethod;

	FrameField () {
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
		csrv_s_ = NULL;
		csrc_s_ = NULL;
		csrr_s_ = NULL;
		nnz_s_ = 0;
		csrv_ltl_ = NULL;
		csrc_ltl_ = NULL;
		csrr_ltl_ = NULL;
		nnz_ltl_ = 0;
		csrv_l_ = NULL;
		csrc_l_ = NULL;
		csrr_l_ = NULL;
		nnz_l_ = 0;
		lambda_ = 1.f;
	}

	/*
	 * Accessors and Mutators
	 */
	float cell_size () const { return cell_size_; }
	const Vec3f & bbox_min () const { return bbox_min_; }
	const Vec3f & bbox_max () const { return bbox_max_; }
	const Vec3i & res () const { return res_; }

	const std::vector<Vec3f> & opengl_frame_field_p () {
		return opengl_frame_field_p_;
	}
	const std::vector<Vec3f> & opengl_frame_field_n () {
		return opengl_frame_field_n_;
	}

	const std::vector<Vec3f> & opengl_hc_frames_p () {
		return opengl_hc_frames_p_;
	}
	const std::vector<Vec3f> & opengl_hc_frames_n () {
		return opengl_hc_frames_n_;
	}

	void NextSliceXY () { if (slice_xy_id_ < res_[2]) slice_xy_id_++; else slice_xy_id_ = 0; }
	void PreviousSliceXY () { if (slice_xy_id_ > 0) slice_xy_id_--; else slice_xy_id_ = (res_[2] - 1); }

	void AddHardConstrainedPoint (const Vec3f & position,
	                              const Vec3f & e_x,
	                              const Vec3f & e_y,
	                              const Vec3f & e_z);
	void AddHardConstrainedPoint (const Vec3f & position,
	                              const Vec3f & twist);
	void UpdateHardConstrainedPoint (int i,
	                                 const Vec3f & e_x,
	                                 const Vec3f & e_y,
	                                 const Vec3f & e_z);
	void UpdateHardConstrainedPoint (int i, const Vec3f & twist);
	Vec3f GetHardConstrainedPointTwist (int i);
	int GetHardConstrainedPointId (const Vec3f & position);
	int GetHardConstrainedPointId (const Vec3f & position, float eps_tolerance);

	const std::vector<FramePoint> & hc_points () const { return hc_points_; }
	std::vector<FramePoint> & hc_points () { return hc_points_; }
	int GetNumberOfHardConstrainedPoints () const { return hc_points_.size (); }

	const float * quats_pong () const { return quats_pong_; }

	/*
	 * Initialization
	 */
	void Init (const Vec3f & bbox_min, const Vec3f & bbox_max,
	           const Vec3i & res, OptimMethod optim_method);
	/*
	 * Local Optimization
	 */
	void InitCUDAGCubeQuats ();
	void InitBiHarmonicMasks ();
	void InitLocalOptimization (const Vec3f & bbox_min, const Vec3f & bbox_max,
	                            const Vec3i & res);
	/*
	 * Sparse Biharmonic System
	 */
	void InitSparseBiharmonicSystem (const Vec3f & bbox_min, const Vec3f & bbox_max,
	                                 const Vec3i & res);
	void SolveLocalOptimization (int num_iters, bool use_group_symmetry);
	void BuildSparseBiharmonicSystem ();
	void BuildSparseLTLMatrix ();
	void BuildSparseSMatrix ();
	void DebugLTLMatrix ();
	void SolveSparseBiharmonicSystem (bool update_only);
	void StrideGPUCopy (float * dst, float * src, int dst_stride_id,
	                    int dst_stride_size, int size);
	void InitializeQuaternions (float * quats, const float4 & init_quat);

	/*
	 * Generation of the full frame field
	 */
	void Generate (bool update_only, bool use_group_symmetry);


	void InitExploreSO3 (const Vec3f & bbox_min, const Vec3f & bbox_max,
	                     const Vec3i & res);
	void ExploreSO3 (int k_x, int k_y, int k_z);

	/*
	 * Utility functions related to SO3 and other symmetry
	 * spaces
	 */
	static Vec3f RotationToTwistStable (const Vec3f & e_x,
	                                    const Vec3f & e_y,
	                                    const Vec3f & e_z);
	static Vec3f RotationToTwistEigenDecomp (const Vec3f & e_x,
	        const Vec3f & e_y,
	        const Vec3f & e_z);
	static Vec3f RotationToTwistCayley (const Vec3f & e_x,
	                                    const Vec3f & e_y,
	                                    const Vec3f & e_z);
	static Vec3d RotationToTwistCayley (const Vec3d & e_x,
	                                    const Vec3d & e_y,
	                                    const Vec3d & e_z);
	static Vec3f RotationToFactoredTwist (const Vec3f & e_x,
	                                      const Vec3f & e_y,
	                                      const Vec3f & e_z);
	static Vec3f RotationToFactoredTwist (const Vec3d & e_x,
	                                      const Vec3d & e_y,
	                                      const Vec3d & e_z);
	static void TwistToRotation (const Vec3f & twist,
	                             Vec3f & e_x, Vec3f & e_y, Vec3f & e_z);
	static Vec3f TwistToFactoredTwist (const Vec3f & twist);
	static Vec3f TwistToSymmetricTwist (const Vec3f & twist,
	                                    int k_x, int k_y, int k_z);

	static Vec3f Rotate (const Vec3f & twist, const Vec3f & v);
	static Vec3d Rotate (const Vec3d & twist, const Vec3d & v);
	static void RotationToSymmetricRotation (const Vec3f & e_x,
	        const Vec3f & e_y,
	        const Vec3f & e_z,
	        int k_x, int k_y, int k_z,
	        Vec3f & e_x_symm,
	        Vec3f & e_y_symm,
	        Vec3f & e_z_symm);
	static void SO3ToQSO3 (const Vec3f & twist,
	                       Vec3f & qs2, Vec3f & qs1);
	static Vec3f QuatToTwist (const float4 & quat);
	static Vec3f QuatToTwist (float q0, float q1, float q2, float q3);

	/*
	 * OpenGL related functions (mostly to visualize the field)
	 */
	void ComputeOpenGLFrameField ();
	void ComputeOpenGLFrameFieldSliceXY ();
	void ComputeOpenGLHardConstrainedFrames ();
	void ComputeOpenGLFrameFieldPingPongInterpol (float alpha);
	void ComputeOpenGLCube ();

protected:

	void print (const std::string & msg);

private:
	std::vector<FramePoint> hc_points_;
	int max_hc_number_;

	Vec3f bbox_min_;
	Vec3f bbox_max_;
	Vec3i res_;
	float cell_size_;

	OptimMethod optim_method_;

	// Frame Field
	std::vector<Vec3f> twist_field_ping_; // Ping buffer
	std::vector<Vec3f> twist_field_pong_; // Pong buffer

	std::vector<float4> quat_field_ping_; // Ping buffer
	std::vector<float4> quat_field_pong_; // Pong buffer

	std::vector<float> w_hc_quats_host_;
	std::vector<float4> hc_quats_host_;

	// Position Field
	std::vector<Vec3f> position_field_ping_;
	std::vector<Vec3f> position_field_pong_;

	// CPU Group Quaternions
	float4 gcube_quats_vec_[24];

	/*
	 * Local Optimization
	 */
	// GPU Pointers
	float * w_hc_quats_;
	float * hc_quats_;
	float * quats_ping_;
	float * quats_pong_;
	float * init_quats_;

	std::vector<float4 *> multi_grid_quats_;
	// CPU BiHarmonic Masks
	std::vector< std::vector<float> > bh_masks_host_;
	float bh_masks_spectral_radius_;
	// GPU BiHarmonic Masks
	float * bh_masks_;

	// Sparse Biharmonic System
	cusparseMatDescr_t descr_bh_;
	float * x_bh_;
	float * rhs_bh_;
	float * csrv_bh_; // Full System Matrix
	int * csrc_bh_;
	int * csrr_bh_;
	int nnz_bh_;
	float * csrv_l_; // Harmonic term
	int * csrc_l_;
	int * csrr_l_;
	int nnz_l_;
	float * csrv_ltl_; // Biharmonic term
	int * csrc_ltl_;
	int * csrr_ltl_;
	int nnz_ltl_;
	float * csrv_s_; // Screen term
	int * csrc_s_;
	int * csrr_s_;
	int nnz_s_;
	int n_bh_;
	float lambda_;
	std::vector<Vec3i> h_mask_ids_;
	std::vector<float> h_mask_vals_;
	std::vector<Vec3i> bh_mask_ids_;
	std::vector<float> bh_mask_vals_;

	// CUDA Libraires Contexts
	unsigned char * cu_csr_schol_p_buffer_;
	csrcholInfo_t cu_csr_schol_info_;
	cusolverSpHandle_t cusolver_handle_;
	cusparseHandle_t cusparse_handle_;
	cudaStream_t cu_stream_;

	// Opengl Frame Field
	int num_tris_in_cube_;
	std::vector<Vec3f> cube_p_;
	std::vector<Vec3f> cube_n_;
	std::vector<Vec3f> opengl_frame_field_p_; // OpenGL ready frame field (with transformed cubes) : positions
	std::vector<Vec3f> opengl_frame_field_n_; // OpenGL ready frame field (with transformed cubes) : normals
	int slice_xy_id_;
	std::vector<Vec3f> opengl_hc_frames_p_; // OpenGL ready hc frames (with transformed cubes) : positions
	std::vector<Vec3f> opengl_hc_frames_n_; // OpenGL ready hc frames (with transformed cubes) : normals

	void CheckCUDAError ();
};


}
#endif   /* ----- #ifndef FrameField_INC  ----- */
