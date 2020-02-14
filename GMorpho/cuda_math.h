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

#ifndef  BROXY_CUDA_MATH_H
#define  BROXY_CUDA_MATH_H

#include <vector>
#include <vector_types.h>
#include <cfloat>
#include <omp.h>

#include <Common/Vec3.h>

#include "timing.h"

//#define GET_TIME  1e3*omp_get_wtime

#define NAN_EVAL 1e20f
#define NAN_EVAL_THRESH 1e10f

#define SLOT_EMPTY 0xffffffff

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#ifdef __CUDACC__

#define SQR4_FLT_EPSILON 0.018581

__host__ __device__
inline  int pow3i (int i) { return i * i * i; }

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__device__
inline  unsigned int ExpandBits (unsigned int v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
inline  unsigned int Morton3Df (float x, float y, float z) {
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = ExpandBits((unsigned int)x);
	unsigned int yy = ExpandBits((unsigned int)y);
	unsigned int zz = ExpandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

/// Calculates a 30-bit Morton code for the given 3D point located within the grid.
__device__
inline  unsigned int Morton3Dui (unsigned int x, unsigned int y, unsigned int z) {
	unsigned int xx = ExpandBits (x);
	unsigned int yy = ExpandBits (y);
	unsigned int zz = ExpandBits (z);
	return xx * 4 + yy * 2 + zz;
}

/// "Insert" a 0 bit after each of the 16 low bits of x
__device__
inline unsigned int Part1By1 (unsigned int x) {
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}

/// "Insert" two 0 bits after each of the 10 low bits of x
__device__
inline unsigned int Part1By2 (unsigned int x) {
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

/// Inverse of Part1By1 - "delete" all odd-indexed bits
__device__
inline unsigned int Compact1By1 (unsigned int x) {
	x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

/// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
__device__
inline unsigned int Compact1By2 (unsigned int x) {
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

__device__
inline unsigned int EncodeMorton2 (unsigned int x, unsigned int y) {
	return (Part1By1 (y) << 1) + Part1By1 (x);
}

__device__
inline unsigned int EncodeMorton3 (unsigned int x,
                                   unsigned int y,
                                   unsigned int z) {
	return (Part1By2 (z) << 2) + (Part1By2 (y) << 1) + Part1By2 (x);
//	return (Part1By2 (x) << 2) + (Part1By2 (y) << 1) + Part1By2 (z);
}

__device__
inline unsigned int DecodeMorton2X(unsigned int code) {
	return Compact1By1 (code >> 0);
}

__device__
inline  unsigned int DecodeMorton2Y(unsigned int code) {
	return Compact1By1 (code >> 1);
}

__device__
inline  unsigned int DecodeMorton3X(unsigned int code) {
	return Compact1By2 (code >> 0);
}

__device__
inline  unsigned int DecodeMorton3Y(unsigned int code) {
	return Compact1By2 (code >> 1);
}

__device__
inline  unsigned int DecodeMorton3Z(unsigned int code) {
	return Compact1By2 (code >> 2);
}

struct IsAContour {
	__host__ __device__
	bool operator() (const unsigned char u) {
		return (u != 0 && u != 255);
	}
};

struct IsNonZero {
	__host__ __device__
	bool operator() (const unsigned char u) {
		return (u != 0);
	}
};

/// Compute 3D indices from linear indices
__host__ __device__
inline void Compute3DIdx (unsigned int key, unsigned int resx, unsigned int resxy, float3 & id) {
	unsigned int remainder;
	id.z = key / resxy;
	remainder = key % resxy;
	id.y = remainder / (resx);
	id.x = remainder % (resx);
}

//// Compute 3D indices from linear indices
__host__
inline  void Compute3DIdx (unsigned int key, unsigned int resx, unsigned int resxy, MorphoGraphics::Vec3ui & id) {
	unsigned int remainder;
	id[2] = key / resxy;
	remainder = key % resxy;
	id[1] = remainder / (resx);
	id[0] = remainder % (resx);
}

/// Compute 3D indices from linear indices
__host__
inline  void Compute3DIdx (unsigned int key, unsigned int resx, unsigned int resxy, MorphoGraphics::Vec3f & id) {
	unsigned int remainder;
	id[2] = key / resxy;
	remainder = key % resxy;
	id[1] = remainder / (resx);
	id[0] = remainder % (resx);
}

/// Constant CUDA declaration of Group Quaternions
#define dCPI4 7.0710678118654757274e-01

__constant__ float4 gcube_quats[24];

__host__ __device__
inline bool TestLimit (const int3 & id3, const int3 & res) {
	return (0 <= id3.x) && (id3.x < res.x)
	       && (0 <= id3.y) && (id3.y < res.y)
	       && (0 <= id3.z) && (id3.z < res.z);
}

__host__ __device__
inline void GridToMask (const int & grid_id, const int & grid_res, const int & mask_ext, int & mask_id) {
	if (grid_id <= mask_ext)
		mask_id = grid_id;
	else if (grid_id  < (grid_res - mask_ext))
		mask_id = mask_ext; // set the id to central coordinate
	else
		mask_id = grid_id - (grid_res - (2 * mask_ext + 1));
}

__host__ __device__
inline void GridToMask (const int3 & grid_id3, const int3 & grid_res3, const int3 & mask_ext3, int3 & mask_id3) {
	GridToMask (grid_id3.x, grid_res3.x, mask_ext3.x, mask_id3.x);
	GridToMask (grid_id3.y, grid_res3.y, mask_ext3.y, mask_id3.y);
	GridToMask (grid_id3.z, grid_res3.z, mask_ext3.z, mask_id3.z);
}

// ---------------------------------------------
// Basic vectorial operator set for CUDA float4.
// ---------------------------------------------

__host__ __device__
inline float4 operator+ (const float4 & a, const float4 & b) {
	float4 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	r.w = a.w + b.w;
	return r;
};

__host__ __device__
inline float4 operator- (const float4 & a, const float4 & b) {
	float4 r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;
	r.w = a.w - b.w;
	return r;
};

__host__ __device__
inline float4 operator* (const float4 & a, const float4 & b) {
	float4 r;
	r.x = a.x * b.x;
	r.y = a.y * b.y;
	r.z = a.z * b.z;
	r.w = a.w * b.w;
	return r;
};

__host__ __device__
inline float4 operator/ (const float4 & a, const float4 & b) {
	float4 r;
	r.x = a.x / b.x;
	r.y = a.y / b.y;
	r.z = a.z / b.z;
	r.w = a.w / b.w;
	return r;
};

__host__ __device__
inline float4 operator* (const float4 & a, float s) {
	float4 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	r.w = a.w * s;
	return r;
};

__host__ __device__
inline float4 operator/ (const float4 & a, float s) {
	float4 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	r.w = a.w / s;
	return r;
};

__host__ __device__
inline float4 operator* (float s, const float4 & a) {
	float4 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	r.w = a.w * s;
	return r;
};

__host__ __device__
inline float4 operator/ (float s, const float4 & a) {
	float4 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	r.w = a.w / s;
	return r;
};

__host__ __device__
inline float length (const float4 & a) {
	return sqrt (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
};

__host__ __device__
inline float4 & normalize (float4 & a) {
	a = a / length (a);
	return a;
};

__host__ __device__
inline float distanceR (const float4 & a, const float4 & b) {
	float4 r = a - b;
	return length (r);
};

__host__ __device__
inline float dotProduct (const float4 & a, const float4 & b) {
	float4 r = a * b;
	return (r.x + r.y + r.z + r.w);
};


__host__ __device__
inline float distanceS (const float4 & a, const float4 & b) {
	float4 r = a - b;
	return dotProduct (r, r);
};

__host__ __device__
inline float4 & conj (float4 & a) {
	a.x = -a.x;
	a.y = -a.y;
	a.z = -a.z;
	return a;
}

__host__ __device__
inline float4 conj (const float4 & a) {
	float4 c;
	c.x = -a.x;
	c.y = -a.y;
	c.z = -a.z;
	c.w = a.w;
	return c;
}

__host__ __device__
inline float4 hamilton (const float4 & a, const float4 & b) {
	float4 r;
	r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
	r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
	r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
	r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
	return r;
};

__host__ __device__
inline float3 rotate (const float4 & rot, const float3 & u) {
	float4 q_u = make_float4 (u.x, u.y, u.z, 0.f);
	float4 q_rot_u = hamilton (conj (rot), hamilton (q_u, rot));
	float3 rot_u = make_float3 (q_rot_u.x, q_rot_u.y, q_rot_u.z);
	return rot_u;
}

__host__ __device__
inline float4 interpolate_cube_group (const float4 & q_0, const float4 & q_1, float t, const float4 group_quat[24]) {
	float4 min_q_0_1_interpol;
	float min_dist_q_0_1_interpol = FLT_MAX;
	float t0 = 1.f - t, t1 = t;
	for (int i = 0; i < 24; i++)
		for (int j = 0; j < 24; j++) {
			float4 q_0_symm = hamilton (q_0, group_quat[i]);
			float4 q_1_symm = hamilton (q_1, group_quat[j]);
			float4 q_0_1_interpol = t0 * q_0_symm + t1 * q_1_symm;
			float dist_q_0_1_interpol = distanceS (q_0_1_interpol, q_0) +
			                            distanceS (q_0_1_interpol, q_1);
			if (dist_q_0_1_interpol < min_dist_q_0_1_interpol) {
				min_dist_q_0_1_interpol = dist_q_0_1_interpol;
				min_q_0_1_interpol = q_0_1_interpol;
			}
		}
	return min_q_0_1_interpol;
}

__host__ __device__
inline float4 interpolate_cube_group (const float4 & q_0, const float4 & q_1, float t, const float4 group_quats[24], int group_quats_size) {
	float4 min_q_0_1_interpol;
	float min_dist_q_0_1_interpol = FLT_MAX;
	float t0 = 1.f - t, t1 = t;

	if (group_quats_size == 0)
		return t0 * q_0 + t1 * q_1;

	for (int i = 0; i < group_quats_size; i++)
		for (int j = 0; j < group_quats_size; j++) {
			float4 q_0_symm = hamilton (q_0, group_quats[i]);
			float4 q_1_symm = hamilton (q_1, group_quats[j]);
			float4 q_0_1_interpol = t0 * q_0_symm + t1 * q_1_symm;
			float dist_q_0_1_interpol = distanceS (q_0_1_interpol, q_0) +
			                            distanceS (q_0_1_interpol, q_1);
			if (dist_q_0_1_interpol < min_dist_q_0_1_interpol) {
				min_dist_q_0_1_interpol = dist_q_0_1_interpol;
				min_q_0_1_interpol = q_0_1_interpol;
			}
		}
	min_q_0_1_interpol = normalize (min_q_0_1_interpol);

	return min_q_0_1_interpol;
}

// ---------------------------------------------
// Basic vectorial operator set for CUDA float3.
// ---------------------------------------------

__host__ __device__
inline float3 operator+ (const float3 & a, const float3 & b) {
	float3 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	return r;
};

__host__ __device__
inline float3 operator- (const float3 & a, const float3 & b) {
	float3 r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;
	return r;
};

__host__ __device__
inline float3 operator* (const float3 & a, const float3 & b) {
	float3 r;
	r.x = a.x * b.x;
	r.y = a.y * b.y;
	r.z = a.z * b.z;
	return r;
};

__host__ __device__
inline float3 operator/ (const float3 & a, const float3 & b) {
	float3 r;
	r.x = a.x / b.x;
	r.y = a.y / b.y;
	r.z = a.z / b.z;
	return r;
};

__device__
inline  float3 operator* (const float3 & a, float s) {
	float3 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	return r;
};

__device__
inline  float3 operator/ (const float3 & a, float s) {
	float3 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	return r;
};

__device__
inline  float3 operator* (float s, const float3 & a) {
	float3 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	return r;
};

__device__
inline  float3 operator/ (float s, const float3 & a) {
	float3 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	return r;
};

__device__
inline  void weightFMA (float3 & s, float w, float3 v) {
	s.x = fma (w, v.x, s.x);
	s.y = fma (w, v.y, s.y);
	s.z = fma (w, v.z, s.z);
}

__device__
inline  float length (const float3 & a) {
	return sqrt (a.x * a.x + a.y * a.y + a.z * a.z);
};

__device__
inline  float3 & normalize (float3 & a) {
	a = a / length (a);
	return a;
};

__device__
inline  float distanceR (const float3 & a, const float3 & b) {
	float3 r = a - b;
	return length (r);
};

__device__
inline  float dotProduct (const float3 & a, const float3 & b) {
	float3 r = a * b;
	return (r.x + r.y + r.z);
};

__device__
inline  float distanceS (const float3 & a, const float3 & b) {
	float3 r = a - b;
	return dotProduct (r, r);
};

__device__
inline  float3 crossProduct (const float3 & a, const float3 & b) {
	float3 r;
	r.x = a.y * b.z - a.z * b.y;
	r.y = a.z * b.x - a.x * b.z;
	r.z = a.x * b.y - a.y * b.x;
	return r;
};

__device__
inline  void qRot (const float3 & vec, const float4 & q, float3 & rotVec) {
	float3 r = make_float3 (q.x, q.y, q.z);
	float w = q.w;
	rotVec = vec + 2.f * crossProduct (r, crossProduct (r, vec) + w * vec);
}

__device__
inline  void expMapToQuat (const float3 & expMap, float4 & q) {
	float theta, a, b;

	theta = length (expMap);
	if (theta > SQR4_FLT_EPSILON)
		a = sin (0.5f * theta) / theta;
	else
		a = 0.5f + theta * theta;

	b = cos (0.5f * theta);
	q.x = a * expMap.x;
	q.y = a * expMap.y;
	q.z = a * expMap.z;
	q.w = b;
}

__device__
inline  void expMapRot (const float3 & vec, const float3 & expMap, float3 & rotVec) {
	float4 q;
	expMapToQuat (expMap, q);
	qRot (vec, q, rotVec);
}


__device__
inline  float normS (const float3 & u, float alpha) {
	if (alpha > 0)
		return pow (pow (fabs (u.x), alpha)
		            + pow (fabs (u.y), alpha)
		            + pow (fabs (u.z), alpha), 2.f / alpha);
	else
		return dotProduct (u, u);
}

__device__
inline  float norm (const float3 & u, float alpha) {
	if (alpha > 0)
		return pow (pow (fabs (u.x), alpha)
		            + pow (fabs (u.y), alpha)
		            + pow (fabs (u.z), alpha), 1.f / alpha);
	else
		return length (u);
}

__device__
inline  float distanceS (const float3 & a, const float3 & b, float alpha, const float3 & expMap) {
	float3 ba, r;
	ba = a - b;
	expMapRot (ba, expMap, r);
	return normS (r, alpha);
};

__device__
inline  float distance (const float3 & a, const float3 & b, float alpha, const float3 & expMap) {
	float3 ba, r;
	ba = a - b;
	expMapRot (ba, expMap, r);
	return norm (r, alpha);
};


__device__
inline  float distanceS (const float3 & a, const float3 & b, float alpha) {
	return normS (a - b, alpha);
};

__device__
inline  float distance (const float3 & a, const float3 & b, float alpha) {
	return norm (a - b, alpha);
};

__device__
inline  float3 projectOn (const float3 & x, const float3 & N, const float3 & C) {
	float w = dotProduct ((x - C), N);
	return (x - (N * w));
}

__host__ __device__
inline float2 operator+ (const float2 & a, const float2 & b) {
	float2 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	return r;
};

__host__ __device__
inline float square (float x) { return x * x; }

__host__ __device__
inline float wendland (float x, float h) {
	x = fabs (x);
	if (x < h)
		return square (square (1 - x / h)) * (4 * x / h + 1);
	else
		return 0.0;
}

__host__ __device__
inline float wendlandHole (float x, float h, float s) {
	x = fabs (x);

	if (x < s)
		return 0.0;
	else if (x > s + h)
		return 1.f;
	else
		return wendland (x - (s + h), h);
}

__host__ __device__
inline float fermi (float x, float mu, float step, float alpha) {
	x = fabs (x);
	float beta = (alpha - 1.f) * (1.f + exp (-mu / step));
	float distrib = 1.f + beta / (exp ((x - mu) / step) + 1.f);
	return distrib;
}

__host__ __device__
inline float tukey (float x, float h, float v) {
	x = fabs (x);
	if (x < v)
		return 1.f;
	else if (x < (v + h))
		return square (1 - square ((x - v) / h));
	else
		return 0.0;
}

__host__ __device__
inline float welsch (float x, float h) {
	x = fabs (x);
	if (x < h)
		return square (1 - square (x / h));
	else
		return 0.0;
}

__host__ __device__
inline float compact (float x,
                      float h) {
	x = fabs (x);
	if (x < h)
		return square (square (1 - x / h));
	else
		return 0.0;
}

__device__
inline  double3 operator+ (const double3 & a, const double3 & b) {
	double3 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	return r;
};

__device__
inline  double3 operator- (const double3 & a, const double3 & b) {
	double3 r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;
	return r;
};

__device__
inline  double3 operator* (const double3 & a, const double3 & b) {
	double3 r;
	r.x = a.x * b.x;
	r.y = a.y * b.y;
	r.z = a.z * b.z;
	return r;
};

__device__
inline  double3 operator/ (const double3 & a, const double3 & b) {
	double3 r;
	r.x = a.x / b.x;
	r.y = a.y / b.y;
	r.z = a.z / b.z;
	return r;
};

__device__
inline  double3 operator* (const double3 & a, double s) {
	double3 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	return r;
};

__device__
inline  double3 operator/ (const double3 & a, double s) {
	double3 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	return r;
};

__device__
inline  double3 operator* (double s, const double3 & a) {
	double3 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	return r;
};

__device__
inline  double3 operator/ (double s, const double3 & a) {
	double3 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	return r;
};

__device__
inline  void weightFMA (double3 & s, double w, double3 v) {
	s.x = fma (w, v.x, s.x);
	s.y = fma (w, v.y, s.y);
	s.z = fma (w, v.z, s.z);
}

__device__
inline  double length (const double3 & a) {
	return sqrt (a.x * a.x + a.y * a.y + a.z * a.z);
};

__device__
inline  double3 & normalize (double3 & a) {
	a = a / length (a);
	return a;
};

__device__
inline  double distanceR (const double3 & a, const double3 & b) {
	double3 r = a - b;
	return length (r);
};

__device__
inline  double dotProduct (const double3 & a, const double3 & b) {
	double3 r = a * b;
	return (r.x + r.y + r.z);
};

__device__
inline  double distanceS (const double3 & a, const double3 & b) {
	double3 r = a - b;
	return dotProduct (r, r);
};

__device__
inline  double distanceS (const double3 & a, const double3 & b, double alpha) {
	double3 r = a - b;
	if (alpha > 0)
		return pow (pow (fabs (r.x), alpha)
		            + pow (fabs (r.y), alpha)
		            + pow (fabs (r.z), alpha), 2.f / alpha);
	else
		return dotProduct (r, r);

};

__device__
inline  double distance (const double3 & a, const double3 & b, double alpha) {
	double3 r = a - b;
	if (alpha > 0)
		return pow (pow (fabs (r.x), alpha)
		            + pow (fabs (r.y), alpha)
		            + pow (fabs (r.z), alpha), 1.f / alpha);
	else
		return length (r);
};

__device__
inline  double3 crossProduct (const double3 & a, const double3 & b) {
	double3 r;
	r.x = a.y * b.z - a.z * b.y;
	r.y = a.z * b.x - a.x * b.z;
	r.z = a.x * b.y - a.y * b.x;
	return r;
};

__device__
inline  double3 projectOn (const double3 & x, const double3 & N, const double3 & C) {
	double w = dotProduct ((x - C), N);
	return (x - (N * w));
}

__host__ __device__
inline uint3 operator+ (const uint3 & a, const uint3 & b) {
	uint3 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	return r;
};

__host__ __device__
inline uint3 operator* (const uint3 & a, const uint3 & b) {
	uint3 r;
	r.x = a.x * b.x;
	r.y = a.y * b.y;
	r.z = a.z * b.z;
	return r;
};

__device__
inline  uint3 operator* (const uint3 & a, float s) {
	uint3 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	return r;
};

__device__
inline  uint3 operator/ (const uint3 & a, float s) {
	uint3 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	return r;
};

__device__
inline  uint3 operator* (float s, const uint3 & a) {
	uint3 r;
	r.x = a.x * s;
	r.y = a.y * s;
	r.z = a.z * s;
	return r;
};

__device__
inline  uint3 operator/ (float s, const uint3 & a) {
	uint3 r;
	r.x = a.x / s;
	r.y = a.y / s;
	r.z = a.z / s;
	return r;
};

__host__ __device__
inline int3 operator+ (const int3 & a, const int3 & b) {
	int3 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	return r;
};

__device__
inline  bool PointAABBIntersect (const float3 & c, const float3 & aabb_min, const float3 & aabb_max) {
	bool intersect = false;
	if (c.x < aabb_min.x)
		intersect = true;
	else if (c.x > aabb_max.x)
		intersect = true;
	if (c.y < aabb_min.y)
		intersect = true;
	else if (c.y > aabb_max.y)
		intersect = true;
	if (c.z < aabb_min.z)
		intersect = true;
	else if (c.z > aabb_max.z)
		intersect = true;

	return intersect;
}

__device__
inline  bool SphereAABBIntersect (const float3 & c, float sq_radius, const float3 & aabb_min, const float3 & aabb_max) {
	float sq_d = 0.f;
	if (c.x < aabb_min.x)
		sq_d += square (c.x - aabb_min.x);
	else if (c.x > aabb_max.x)
		sq_d += square (c.x - aabb_max.x);
	if (c.y < aabb_min.y)
		sq_d += square (c.y - aabb_min.y);
	else if (c.y > aabb_max.y)
		sq_d += square (c.y - aabb_max.y);
	if (c.z < aabb_min.z)
		sq_d += square (c.z - aabb_min.z);
	else if (c.z > aabb_max.z)
		sq_d += square (c.z - aabb_max.z);

	return (sq_d < sq_radius);
}

__device__
inline  float SphereAABBSqDistance (const float3 & c, const float3 & aabb_min, const float3 & aabb_max) {
	float sq_d = 0.f;
	if (c.x < aabb_min.x)
		sq_d += square (c.x - aabb_min.x);
	else if (c.x > aabb_max.x)
		sq_d += square (c.x - aabb_max.x);
	if (c.y < aabb_min.y)
		sq_d += square (c.y - aabb_min.y);
	else if (c.y > aabb_max.y)
		sq_d += square (c.y - aabb_max.y);
	if (c.z < aabb_min.z)
		sq_d += square (c.z - aabb_min.z);
	else if (c.z > aabb_max.z)
		sq_d += square (c.z - aabb_max.z);

	return sq_d;
}

__device__
inline  bool SpherePointIntersect (const float3 & c, float sq_radius, const float3 & p) {
	float3 cp = c - p;
	float sq_d = dotProduct (cp, cp);
	return (sq_d < sq_radius);
}

#endif


#endif // BROXY_CUDA_MATH_H