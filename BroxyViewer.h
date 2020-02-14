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

#ifndef BROXYVIEWER_H
#define BROXYVIEWER_H

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <QString>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QDir>
#include <string>
#include <vector>

#include <cfloat>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/OrderingMethods>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>

#include <Common/Mesh.h>
#include <Common/OpenGL.h>

#include "ScaleField.h"

class BroxyViewer : public QGLViewer  {
	Q_OBJECT

public:
	BroxyViewer (QWidget * parent = 0);
	virtual ~BroxyViewer ();

	void featureAwareDecimation (bool on_morpho_output);
	void generateFrameField (bool update_only, bool use_group_symmetry);
	MorphoGraphics::Mesh & morpho_mesh () { return _morpho_mesh; }
	MorphoGraphics::ScaleField & scale_field () { return _scale_field; }

	inline bool isDisplaySmoothShading () const  { return _drawSmooth; }
	inline bool isDisplayInputMesh () const  { return _drawInputMesh; }
	inline bool isDisplayTransparentInputMesh () const  { return _isInputMeshTransparent; }
	inline bool isDisplayProxy () const  { return _drawMorphoMesh; }
	inline bool isDisplayTransparentProxy () const  { return _isMorphoMeshTransparent; }
	inline bool isDisplayProxyMesh () const  { return _drawCageMesh; }
	inline bool isDisplayTransparentProxyMesh () const  { return _isCageMeshTransparent; }

public slots:
	void loadMesh (const QString & filename);
	void storeProxy (const QString & filename);
	void toggleAsymmetricClosing (bool b);
	void toggleRotationField (bool b);
	void increaseBaseScale ();
	void decreaseBaseScale ();

	void toggleDisplaySmoothShading (bool b);
	void toggleDisplayInputMesh (bool b);
	void toggleDisplayTransparentInputMesh (bool b);
	void toggleDisplayProxy (bool b);
	void toggleDisplayTransparentProxy (bool b);
	void toggleDisplayProxyMesh (bool b);
	void toggleDisplayTransparentProxyMesh (bool b);

protected :
	void init();
	void draw ();
	void resizeGL (int width, int height);
	QString helpString() const;
	void keyPressEvent (QKeyEvent * e);
	void mousePressEvent (QMouseEvent * e);
	void mouseMoveEvent (QMouseEvent * e);
	//		void turnTableAnimation ();
	virtual void animate ();

private:
	MorphoGraphics::Mesh _mesh;
	MorphoGraphics::Mesh _morpho_mesh;
	MorphoGraphics::Mesh _cage;
	std::vector<MorphoGraphics::Vec3f> _frames_tri_p;
	std::vector<MorphoGraphics::Vec3f> _frames_tri_n;
	int _slice_xy_id;
	int _frames_tri_start;
	int _frames_tri_slice_size;
	int _frames_tri_num;
	
	std::vector<MorphoGraphics::Vec3f> _hc_frames_tri_p;
	std::vector<MorphoGraphics::Vec3f> _hc_frames_tri_n;
	std::vector<MorphoGraphics::Vec3f> _hc_frames_pos;
	std::vector<MorphoGraphics::Vec3f> _hc_frames_twist;
	int _curr_frame;
	MorphoGraphics::Vec3f _curr_twist;
	MorphoGraphics::Vec3f _drag_start;
	MorphoGraphics::Vec3f _drag_end;
	int _curr_i;
	int _curr_symm;

	// **************************************************************************
	// Cage extras for quadmeshing and flat rendering
	// **************************************************************************
	std::vector<unsigned int> _triQuadCageIndices;
	std::vector<std::vector<unsigned int> > _triQuads;
	std::vector<MorphoGraphics::Vec3f> _cageTriNormals; // Update this each time you change the cage.
	void updateCageExtras ();

	// **************************************************************************
	// GPU Shaders
	// **************************************************************************
	MorphoGraphics::GL::Program * _meshProgram;
	MorphoGraphics::GL::Program * _cageProgram;
	MorphoGraphics::GL::Program * _transparentProgram;
	MorphoGraphics::GL::Program * _transparentFinalProgram;
	MorphoGraphics::GL::Program * _scale_fieldProgram;
	
	unsigned int _width;
	unsigned int _height;

	GLuint _aBuffer;
	GLuint _depthTex;
	GLuint _opaqueTex;
	GLuint _accumTex;
	GLuint _revealTex;

	// OpengGL structures management
	class TexTarget {
	public:
		inline TexTarget (GLenum target, GLuint texture) : _target (target), _texture (texture) {}
		inline GLenum target () const { return _target; }
		inline GLuint texture () const { return _texture; }
	private:
		GLenum _target;
		GLuint _texture;
	};
	GLuint genTexture (GLuint & tex,
		GLuint width,
		GLuint height,
		GLint minFilter,
		GLint magFilter,
		GLint internalFormat,
		GLenum format,
		GLenum type);
	void deleteTexture (GLuint & tex);
	GLuint genFramebuffer (GLuint & buffer, const std::vector<TexTarget> & texTargets);
	void deleteFramebuffer (GLuint & buffer);		

	// OpengGL Rendering
	void initTransparencyBuffers ();
	void drawScalePoints ();
	void drawFullScreenQuad ();

	bool _isInputMeshTransparent;
	bool _drawInputMesh;
	bool _isMorphoMeshTransparent;
	bool _drawMorphoMesh;
	bool _isCageMeshTransparent;
	bool _drawCageMesh;
	bool _drawSmooth;
	bool _bilateralMorphoMeshNormals;
	bool _drawQuads;
	bool _drawScaleField;
	bool _drawFrameField;
	bool _isFrameFieldTransparent;

	MorphoGraphics::EditMode _scaleEditMode;
	MorphoGraphics::ScaleField _scale_field;
	MorphoGraphics::Mesh _scalePointMesh;

	bool _takeScreenShot;
	float curr_cam_theta_;
};

#endif // BROXYVIEWER_H

// Some Emacs-Hints -- please don't remove:
//
//  Local Variables:
//  mode:C++
//  tab-width:4
//  End:
