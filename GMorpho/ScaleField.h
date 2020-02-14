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

#ifndef  ScaleField_INC
#define  ScaleField_INC

#include <vector>
#include <Common/Vec3.h>


namespace MorphoGraphics {
typedef enum {NO_EDITION, MAX_BRUSH, MIN_BRUSH, ZERO_BRUSH} EditMode;
class ScaleField {
	class Exception {
	public:
		inline Exception (const std::string & msg) : msg_ ("ScaleField Error: " + msg) {}
		inline const std::string & msg () const { return msg_; }
	protected:
		std::string msg_;
	};

	class ScalePoint {
	public:
		ScalePoint () {

		}
		ScalePoint (const MorphoGraphics::Vec3f & position, float scale) {
			position_ = position;
			scale_ = scale;
			support_ = scale;
			edit_mode_ = MAX_BRUSH;
		}
		ScalePoint (const MorphoGraphics::Vec3f & position, float scale, float support) {
			position_ = position;
			scale_ = scale;
			support_ = support;
			edit_mode_ = MAX_BRUSH;
		}

		~ScalePoint () {

		}
		inline const MorphoGraphics::Vec3f & position () const { return position_; }
		inline MorphoGraphics::Vec3f & position () { return position_; }
		inline float scale () const { return scale_; }
		inline float support () const { return support_; }
		inline EditMode edit_mode () const { return edit_mode_; }
		inline void set_position (const MorphoGraphics::Vec3f & position) { position_ = position; }
		inline void set_scale (float scale) { scale_ = scale; }
		inline void set_support (float support) { support_ = support; }
		inline void set_edit_mode (EditMode edit_mode) { edit_mode_ = edit_mode; }
	private:
		MorphoGraphics::Vec3f position_;
		float scale_;
		float support_;
		EditMode edit_mode_;
	};
public:
	ScaleField () {
		scale_grid_ = NULL;
	}
	float cell_size () const { return cell_size_; }
	const MorphoGraphics::Vec3f & bbox_min () const { return bbox_min_; }
	const MorphoGraphics::Vec3f & bbox_max () const { return bbox_max_; }
	const MorphoGraphics::Vec3<unsigned int> & res () const { return res_; }
	float global_scale () const { return global_scale_; }
	void set_global_scale (float global_scale) {
		global_scale_ = global_scale;
	}
	virtual ~ScaleField () {

	}
	void AddPoint (const MorphoGraphics::Vec3f & position, float scale) {
		ScalePoint s_point (position, scale);
		points_.push_back (s_point);
	}
	void AddPoint (const MorphoGraphics::Vec3f & position, float scale, float support) {
		ScalePoint s_point (position, scale, support);
		points_.push_back (s_point);
	}
	const std::vector<ScalePoint> & points () const {
		return points_;
	}
	std::vector<ScalePoint> & points () {
		return points_;
	}
	int GetNumberOfPoints () const { return points_.size (); }

	void set_bbox_min (const MorphoGraphics::Vec3f & bbox_min) { bbox_min_ = bbox_min; }
	void set_bbox_max (const MorphoGraphics::Vec3f & bbox_max) { bbox_min_ = bbox_max; }
	void set_res (const MorphoGraphics::Vec3<unsigned int> & res) { res_ = res; }
	char * scale_grid () const { return scale_grid_; }

	void GetScaleGridCPU (char ** scale_grid_cpu_ptr);
	void Init (const MorphoGraphics::Vec3f & bbox_min, const MorphoGraphics::Vec3f & bbox_max,
	           const MorphoGraphics::Vec3<unsigned int> & res, float cell_size);

	void UpdateGrid ();
	void UpdateGridGlobalScale ();
protected:
	void ShowGPUMemoryUsage ();
	void print (const std::string & msg);
private:
	std::vector<ScalePoint> points_;
	float global_scale_;

	MorphoGraphics::Vec3f bbox_min_;
	MorphoGraphics::Vec3f bbox_max_;
	MorphoGraphics::Vec3<unsigned int> res_;
	float cell_size_;

	// --------------------------------------------------------------
	//  GPU Data
	// --------------------------------------------------------------
	char * scale_grid_;

	void CheckCUDAError ();
	template<typename T>
	void FreeGPUResource (T ** res);
};
}

#endif   /* ----- #ifndef ScaleField_INC  ----- */
