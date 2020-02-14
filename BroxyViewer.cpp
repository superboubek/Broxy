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

#include "BroxyViewer.h"

#include <queue>
#include <iostream>
#include <QDebug>
#include <QApplication>

#include <Common/Mat4.h>
#include <Common/Ray.h>
#include <Common/BoundingVolume.h>

#include "Voxelizer.h"
#include "GMorpho.h"
#include "FrameField.h"
#include "Decimator.h"

using namespace std;
using namespace MorphoGraphics;

Voxelizer voxelizer;
GMorpho gmorpho; // Morphological Framework
Mesh morpho_mesh;
FrameField frame_field;

static const float MAX_SE_SIZE = 0.3f;
static const float INITIAL_SE_SIZE = 0.01f;
static const float DIST_TO_CENTER = 1.7f;
static const int ROT_AXIS = 1;
static const unsigned int VOXEL_RESOLUTION = 256;
static const qglviewer::Vec UP_VECTOR (0, 1, 0);

BroxyViewer::BroxyViewer (QWidget * parent) : QGLViewer (parent) {
	// Rendering State Initialization
	_width = 1000;
	_height = 1000;
	_isInputMeshTransparent = false;
	_drawInputMesh = true;
	_isMorphoMeshTransparent = true;
	_drawMorphoMesh = true;
	_isCageMeshTransparent = true;
	_drawCageMesh = false;
	_drawSmooth = true;
	_bilateralMorphoMeshNormals = true;
	_drawQuads = false;
	_drawScaleField = false;
	_drawFrameField = false;
	_isFrameFieldTransparent = false;
	_frames_tri_start = 0;
}

BroxyViewer::~BroxyViewer () {

}

void BroxyViewer::drawScalePoints () {
	glDepthMask (GL_FALSE);
	for (int i = 0; i < _scale_field.GetNumberOfPoints (); i++) {
		glUniform3f (_scale_fieldProgram->getUniformLocation ("scalePointPosition"),
		             _scale_field.points ()[i].position ()[0],
		             _scale_field.points ()[i].position ()[1],
		             _scale_field.points ()[i].position ()[2]);
		glUniform1f (_scale_fieldProgram->getUniformLocation ("scaleAlpha"),
		             0.3f);

		glUniform1i (_scale_fieldProgram->getUniformLocation ("cullMode"),
		             0);
		glUniform1f (_scale_fieldProgram->getUniformLocation ("scalePointScale"),
		             _scale_field.points ()[i].support ());
		_scalePointMesh.draw (Mesh::SMOOTH_RENDERING_MODE);
		glUniform1f (_scale_fieldProgram->getUniformLocation ("scalePointScale"),
		             _scale_field.points ()[i].scale ());
		_scalePointMesh.draw (Mesh::SMOOTH_RENDERING_MODE);

		glUniform1i (_scale_fieldProgram->getUniformLocation ("cullMode"),
		             1);
		glUniform1f (_scale_fieldProgram->getUniformLocation ("scalePointScale"),
		             _scale_field.points ()[i].scale ());
		_scalePointMesh.draw (Mesh::SMOOTH_RENDERING_MODE);
		glUniform1f (_scale_fieldProgram->getUniformLocation ("scalePointScale"),
		             _scale_field.points ()[i].support ());
		_scalePointMesh.draw (Mesh::SMOOTH_RENDERING_MODE);
	}
	glDepthMask (GL_TRUE);
}

void BroxyViewer::loadMesh (const QString & filename) {
	try {
		_mesh.load (qPrintable (filename));

		BoundingSphere bs (_mesh.P()[0]);
		for (unsigned int i = 1; i < _mesh.P().size (); i++)
			bs.extendTo (_mesh.P()[i]);
		for (unsigned int i = 0; i < _cage.P().size (); i++)
			bs.extendTo (_cage.P()[i]);
		setSceneCenter (qglviewer::Vec (bs.center()[0], bs.center()[1], bs.center()[2]));
		setSceneRadius (bs.radius());
		showEntireScene ();

		const std::vector<Vec3f> & positions = _mesh.P ();
		const std::vector<Vec3f> & normals = _mesh.N ();
		const std::vector< Vec3<unsigned int> > & triangles = _mesh.T ();
		gmorpho.set_bilateral_filtering (_bilateralMorphoMeshNormals);
		gmorpho.Load (&positions[0][0], &normals[0][0], positions.size (),
		              &triangles[0][0], triangles.size (), VOXEL_RESOLUTION, // VOXEL RESOLUTION
		              MAX_SE_SIZE);

		// Scale Field Initialization
		Vec3f bbox_min = gmorpho.bbox ().min();
		Vec3f bbox_max = gmorpho.bbox ().max();
		Vec3ui res = ((unsigned int)2) * gmorpho.res ();
		float cell_size = 0.5f * gmorpho.cell_size ();
		_scale_field.set_global_scale (INITIAL_SE_SIZE);
		_scale_field.Init (bbox_min, bbox_max, res, cell_size);
		_scale_field.UpdateGridGlobalScale ();
		gmorpho.set_use_asymmetric_closing (true);
		gmorpho.set_use_frame_field (false);
		//	gmorpho.set_use_asymmetric_closing (false);

		// Frame Field Initialization
		bool update_only = false;
		bool use_group_symmetry = false;
		Vec3i frame_field_res (gmorpho.res ()[0] / 16,
		                       gmorpho.res ()[1] / 16,
		                       gmorpho.res ()[2] / 16);
		std::cout << "frame_field_res : " << frame_field_res << std::endl;
		frame_field.Init (bbox_min, bbox_max, frame_field_res,
		                  FrameField::BiHarmonicSystem
		                  //										FrameField::LocalOptimization
		                 );
		frame_field.Generate (update_only, use_group_symmetry);

		// Compute the morphological transformation from
		// the Scale Field and the Frame Field
		gmorpho.Update (_morpho_mesh, _scale_field, frame_field);
	} catch (Mesh::Exception & e) {
		std::cerr << e.msg () << std::endl;
		exit (1);
	}
	updateCageExtras ();

	frame_field.ComputeOpenGLFrameFieldSliceXY ();
	frame_field.ComputeOpenGLHardConstrainedFrames ();
	_curr_frame = -1;

	_frames_tri_p = frame_field.opengl_frame_field_p ();
	_frames_tri_n = frame_field.opengl_frame_field_n ();
	_hc_frames_tri_p = frame_field.opengl_hc_frames_p ();
	_hc_frames_tri_n = frame_field.opengl_hc_frames_n ();

	updateGL ();
}

void BroxyViewer::storeProxy (const QString & filename) {
	_cage.store (qPrintable (filename));
}

void BroxyViewer::toggleAsymmetricClosing (bool b) {
	gmorpho.set_use_asymmetric_closing (b);
	double time1 = GET_TIME ();
	gmorpho.Update (_morpho_mesh, _scale_field, frame_field);
	double time2 = GET_TIME ();
	std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
	updateGL ();
}

void BroxyViewer::toggleRotationField (bool b) {
	gmorpho.set_use_frame_field (b);
	double time1 = GET_TIME ();
	gmorpho.Update (_morpho_mesh, _scale_field, frame_field);
	double time2 = GET_TIME ();
	std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
	updateGL ();
}

void BroxyViewer::increaseBaseScale () {
	double time1, time2;
	float global_scale = _scale_field.global_scale ();
	global_scale += 0.5f * gmorpho.cell_size ();

	global_scale = std::max (global_scale,
	                         0.5f * gmorpho.cell_size ());
	global_scale = std::min (global_scale,
	                         MAX_SE_SIZE);
	if ((global_scale != MAX_SE_SIZE) && (global_scale != 0.5f * gmorpho.cell_size ())) {
		_scale_field.set_global_scale (global_scale);
		time1 = GET_TIME ();
		_scale_field.UpdateGridGlobalScale ();
		gmorpho.Update (_morpho_mesh, _scale_field, frame_field);
		updateGL ();
		time2 = GET_TIME ();
		std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
	}
}

void BroxyViewer::decreaseBaseScale () {
	double time1, time2;
	float global_scale = _scale_field.global_scale ();
	global_scale -= 0.5f * gmorpho.cell_size ();

	global_scale = std::max (global_scale,
	                         0.5f * gmorpho.cell_size ());
	global_scale = std::min (global_scale,
	                         MAX_SE_SIZE);
	if ((global_scale != MAX_SE_SIZE) && (global_scale != 0.5f * gmorpho.cell_size ())) {
		_scale_field.set_global_scale (global_scale);
		time1 = GET_TIME ();
		_scale_field.UpdateGridGlobalScale ();
		gmorpho.Update (_morpho_mesh, _scale_field, frame_field);
		updateGL ();
		time2 = GET_TIME ();
		std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
	}
}

void BroxyViewer::drawFullScreenQuad () {
	const GLfloat quadV[] = { -1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 1.f, 1.f, 0.f, -1.f, 1.f, 0.f};
	const GLuint quadT[] = {0, 1, 2, 0, 2, 3};
	glEnableVertexAttribArray (0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof (float), (GLvoid*)quadV);
	glDrawElements (GL_TRIANGLES, 6, GL_UNSIGNED_INT, (GLvoid*)quadT);
	glDisableVertexAttribArray(0);
}

void BroxyViewer::toggleDisplaySmoothShading (bool b) {
	_drawSmooth = b;
	updateGL ();
}

void BroxyViewer::toggleDisplayInputMesh (bool b) {
	_drawInputMesh = b;
	updateGL ();
}

void BroxyViewer::toggleDisplayTransparentInputMesh (bool b) {
	_isInputMeshTransparent = b;
	updateGL ();
}

void BroxyViewer::toggleDisplayProxy (bool b) {
	_drawMorphoMesh = b;
	updateGL ();
}

void BroxyViewer::toggleDisplayTransparentProxy (bool b) {
	_isMorphoMeshTransparent = b;
	updateGL ();
}

void BroxyViewer::toggleDisplayProxyMesh (bool b) {
	_drawCageMesh = b;
	updateGL ();
}

void BroxyViewer::toggleDisplayTransparentProxyMesh (bool b) {
	_isCageMeshTransparent = b;
	updateGL ();
}

GLuint BroxyViewer::genTexture (GLuint & tex,
                                GLuint width,
                                GLuint height,
                                GLint minFilter,
                                GLint magFilter,
                                GLint internalFormat,
                                GLenum format,
                                GLenum type) {
	deleteTexture (tex);
	glGenTextures (1, &tex);
	glBindTexture (GL_TEXTURE_2D, tex);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D (GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, NULL);
	glBindTexture (GL_TEXTURE_2D, 0);
	return tex;
}

void BroxyViewer::deleteTexture (GLuint & tex) {
	if (tex != 0)
		glDeleteTextures (1, &tex);
	tex = 0;
}

GLuint BroxyViewer::genFramebuffer (GLuint & buffer,
                                    const std::vector<TexTarget> & texTargets) {
	deleteFramebuffer (buffer);
	glGenFramebuffers (1, &buffer);
	glBindFramebuffer (GL_FRAMEBUFFER, buffer);
	for (unsigned int i = 0; i < texTargets.size (); i++)
		glFramebufferTexture2D (GL_FRAMEBUFFER,
		                        texTargets[i].target (),
		                        GL_TEXTURE_2D,
		                        texTargets[i].texture (), 0);
	//	GLenum status = glCheckFramebufferStatus (GL_FRAMEBUFFER);
	//	switch (status) {
	//		case GL_FRAMEBUFFER_COMPLETE:
	//			break;
	//		default:
	//			throw Exception ("Error initializing SSR.");
	//	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return buffer;
}

void BroxyViewer::deleteFramebuffer (GLuint & buffer) {
	if (buffer != 0)
		glDeleteFramebuffers (1, &buffer);
	buffer = 0;
}

void BroxyViewer::initTransparencyBuffers () {
	std::vector<TexTarget> aTexTargets;
	aTexTargets.push_back (TexTarget (GL_DEPTH_ATTACHMENT, genTexture (_depthTex,
	                                  _width,
	                                  _height,
	                                  GL_NEAREST,
	                                  GL_NEAREST,
	                                  GL_DEPTH_COMPONENT24,
	                                  GL_DEPTH_COMPONENT,
	                                  GL_UNSIGNED_BYTE)));

	aTexTargets.push_back (TexTarget (GL_COLOR_ATTACHMENT0, genTexture (_opaqueTex,
	                                  _width,
	                                  _height,
	                                  GL_NEAREST,
	                                  GL_NEAREST,
	                                  GL_RGBA32F,
	                                  GL_RGBA,
	                                  GL_FLOAT)));

	aTexTargets.push_back (TexTarget (GL_COLOR_ATTACHMENT1, genTexture (_accumTex,
	                                  _width,
	                                  _height,
	                                  GL_NEAREST,
	                                  GL_NEAREST,
	                                  GL_RGBA32F,
	                                  GL_RGBA,
	                                  GL_FLOAT)));
	aTexTargets.push_back (TexTarget (GL_COLOR_ATTACHMENT2, genTexture (_revealTex,
	                                  _width,
	                                  _height,
	                                  GL_NEAREST,
	                                  GL_NEAREST,
	                                  GL_RGBA32F,
	                                  GL_RGBA,
	                                  GL_FLOAT)));
	genFramebuffer (_aBuffer, aTexTargets);
}

void BroxyViewer::draw () {
	GLfloat cameraViewMatrix[16];
	glGetFloatv (GL_MODELVIEW_MATRIX, cameraViewMatrix);
	Mat4f cameraNormalMat (cameraViewMatrix);
	cameraNormalMat.invert().transpose();
	GLfloat cameraProjectionMatrix[16];
	glGetFloatv (GL_PROJECTION_MATRIX, cameraProjectionMatrix);
	glLoadIdentity ();

	// Draw Transparent Objects
	_transparentProgram->use();
	glBindFramebuffer (GL_DRAW_FRAMEBUFFER, _aBuffer);
	glClear (GL_DEPTH_BUFFER_BIT);
	const GLfloat opaqueInit[4] = {1.f, 1.f, 1.f, 0.f};
	const GLfloat accumInit[4] = {0.f, 0.f, 0.f, 0.f};
	const GLfloat revealInit[4] = {1.f, 1.f, 1.f, 1.f};
	glClearBufferfv (GL_COLOR, 0, opaqueInit);
	glClearBufferfv (GL_COLOR, 1, accumInit);
	glClearBufferfv (GL_COLOR, 2, revealInit);
	GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
	glDrawBuffers (3, buffers);
	glBlendFunci (0, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendFunci (1, GL_ONE, GL_ONE);
	glBlendFunci (2, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
	glUniformMatrix4fv (_transparentProgram->getUniformLocation ("cameraViewMatrix"), 1, GL_FALSE, cameraViewMatrix);
	glUniformMatrix4fv (_transparentProgram->getUniformLocation ("cameraNormalMatrix"), 1, GL_FALSE, cameraNormalMat.data ());
	glUniformMatrix4fv (_transparentProgram->getUniformLocation ("cameraProjectionMatrix"), 1, GL_FALSE, cameraProjectionMatrix);
	glUniform3f (_transparentProgram->getUniformLocation ("translation"), 0.f, 0.f, 0.f);
	glUniform1f (_transparentProgram->getUniformLocation ("scale"), 1.f);

	glUniform1i (_transparentProgram->getUniformLocation ("isTransparent"), 0);
	if (_drawInputMesh && !_isInputMeshTransparent) {
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 0);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 1);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 0.72f);
		_mesh.draw (Mesh::SMOOTH_RENDERING_MODE);
	}
	if (_drawMorphoMesh && !_isMorphoMeshTransparent) {
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 0);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 1);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 0.72f);
		_morpho_mesh.draw (Mesh::SMOOTH_RENDERING_MODE);
	}

	if (_drawFrameField && !_isFrameFieldTransparent) {
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 0);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 1);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 0.72f);
		glEnableVertexAttribArray (0);
		glEnableVertexAttribArray (1);
		if (_frames_tri_p.size () != 0) {
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(&_frames_tri_p[0]));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(((float*)&_frames_tri_n[0])));
			glDrawArrays (GL_TRIANGLES, 0, _frames_tri_p.size ());
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	if (_drawFrameField && _isFrameFieldTransparent) {
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 0);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 1);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 0.72f);
		glEnableVertexAttribArray (0);
		glEnableVertexAttribArray (1);
		if (_hc_frames_tri_p.size () != 0) {
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(&_hc_frames_tri_p[0]));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(((float*)&_hc_frames_tri_n[0])));
			glDrawArrays (GL_TRIANGLES, 0, _hc_frames_tri_p.size ());
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	if (_drawCageMesh && !_isCageMeshTransparent) {
		if (!_drawQuads) {
			glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 5);
			glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
			glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
			_cage.draw (Mesh::WIRE_RENDERING_MODE);
		}
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 0);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 1);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 0.72f);

		glEnable (GL_POLYGON_OFFSET_FILL);
		glPolygonOffset (1.f, 1.f);
		if (_triQuads.size () != 0 && _drawQuads) {
			_cage.draw (_triQuads, Mesh::FLAT_RENDERING_MODE);
		} else {
			_cage.draw (Mesh::FLAT_RENDERING_MODE);
		}
		glDisable (GL_POLYGON_OFFSET_FILL);

		if (_drawQuads) {
			glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 5);
			glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
			glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
			if (_triQuadCageIndices.size () != 0) {
				glBegin (GL_LINES);
				for (int i = 0; i < (int)(_triQuadCageIndices.size () / 2); i++) {
					glVertex3f (_cage.P ()[_triQuadCageIndices[2 * i]][0],
					            _cage.P ()[_triQuadCageIndices[2 * i]][1],
					            _cage.P ()[_triQuadCageIndices[2 * i]][2]);
					glVertex3f (_cage.P ()[_triQuadCageIndices[2 * i + 1]][0],
					            _cage.P ()[_triQuadCageIndices[2 * i + 1]][1],
					            _cage.P ()[_triQuadCageIndices[2 * i + 1]][2]);
				}
				glEnd ();
				//				glEnableVertexAttribArray (0);
				//				glEnableVertexAttribArray (1);
				//				glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(&(_cage.P()[0])));
				//				glVertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(&(_cage.N()[0])));
				//				glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
				//				glDrawElements (GL_LINES, _triQuadCageIndices.size (), GL_UNSIGNED_INT, (GLvoid*)(&_triQuadCageIndices[0]));
				//				glDisableVertexAttribArray (0);
				//				glDisableVertexAttribArray (1);
			}
		}
	}

	glDepthMask (GL_FALSE);
	glUniform1i (_transparentProgram->getUniformLocation ("isTransparent"),
	             1);
	if (_drawInputMesh && _isInputMeshTransparent) {
		glUniform1f (_transparentProgram->getUniformLocation ("alphaValue"), 0.5f);
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 0);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 1);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 0.72f);
		_mesh.draw (Mesh::SMOOTH_RENDERING_MODE);
	}
	if (_drawMorphoMesh && _isMorphoMeshTransparent) {
		glUniform1f (_transparentProgram->getUniformLocation ("alphaValue"), 0.5f);
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 1);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
		_morpho_mesh.draw (Mesh::SMOOTH_RENDERING_MODE);
	}

	if (_drawFrameField && _isFrameFieldTransparent) {
		glUniform1f (_transparentProgram->getUniformLocation ("alphaValue"), 0.5f);
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 1);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
		glEnableVertexAttribArray (0);
		glEnableVertexAttribArray (1);
		if (_frames_tri_p.size () != 0) {
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(&_frames_tri_p[0]));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(((float*)&_frames_tri_n[0])));
			glDrawArrays (GL_TRIANGLES, 0, _frames_tri_p.size ());
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	if (_drawFrameField && !_isFrameFieldTransparent) {
		glUniform1f (_transparentProgram->getUniformLocation ("alphaValue"), 0.5f);
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 1);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
		glEnableVertexAttribArray (0);
		glEnableVertexAttribArray (1);
		if (_hc_frames_tri_p.size () != 0) {
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(&_hc_frames_tri_p[0]));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f),
			                      (GLvoid*)(((float*)&_hc_frames_tri_n[0])));
			glDrawArrays (GL_TRIANGLES, 0, _hc_frames_tri_p.size ());
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	if (_drawCageMesh && _isCageMeshTransparent) {
		glUniform1f (_transparentProgram->getUniformLocation ("alphaValue"), 0.9f);
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 1);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
		if (_triQuadCageIndices.size () != 0) {
			glEnableVertexAttribArray (0);
			glEnableVertexAttribArray (1);
			glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(&(_cage.P()[0])));
			glVertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(&(_cage.N()[0])));
			glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
			glDrawElements (GL_LINES, _triQuadCageIndices.size (), GL_UNSIGNED_INT, (GLvoid*)(&_triQuadCageIndices[0]));
			glDisableVertexAttribArray (0);
			glDisableVertexAttribArray (1);
		}
		glUniform1f (_transparentProgram->getUniformLocation ("alphaValue"), 0.5f);
		glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 1);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
		_cage.draw (Mesh::FLAT_RENDERING_MODE);
	}
	if (_scaleEditMode != NO_EDITION && _drawScaleField) {
		glUniform1f (_transparentProgram->getUniformLocation ("alphaValue"), 0.5f);
		if (_scaleEditMode == MAX_BRUSH)
			glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 2);
		else if (_scaleEditMode == MIN_BRUSH)
			glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 3);
		else if (_scaleEditMode == ZERO_BRUSH)
			glUniform1i (_transparentProgram->getUniformLocation ("lightMode"), 4);
		glUniform1i (_transparentProgram->getUniformLocation ("metallic"), 0);
		glUniform1f (_transparentProgram->getUniformLocation ("roughness"), 1.4f);
		for (int i = 0; i < _scale_field.GetNumberOfPoints (); i++) {
			glUniform3f (_transparentProgram->getUniformLocation ("translation"),
			             _scale_field.points ()[i].position ()[0],
			             _scale_field.points ()[i].position ()[1],
			             _scale_field.points ()[i].position ()[2]);
			glUniform1f (_transparentProgram->getUniformLocation ("scale"),
			             _scale_field.points ()[i].support ());
			_scalePointMesh.draw (Mesh::SMOOTH_RENDERING_MODE);
			glUniform1f (_transparentProgram->getUniformLocation ("scale"),
			             _scale_field.points ()[i].scale ());
			if (_scaleEditMode != ZERO_BRUSH)
				_scalePointMesh.draw (Mesh::SMOOTH_RENDERING_MODE);
		}
	}
	glDepthMask (GL_TRUE);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);
	glBindFramebuffer (GL_READ_FRAMEBUFFER, _aBuffer);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor (1.f, 1.f, 1.f, 0.f);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_transparentFinalProgram->use ();
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	gluOrtho2D(-1.f, 1.f, -1.f, 1.f);
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity ();
	glGetFloatv (GL_MODELVIEW_MATRIX, cameraViewMatrix);
	glGetFloatv (GL_PROJECTION_MATRIX, cameraProjectionMatrix);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity ();
	try {
		_transparentFinalProgram->setUniformMatrix4fv ("screenViewMatrix", cameraViewMatrix);
		_transparentFinalProgram->setUniformMatrix4fv ("screenProjectionMatrix", cameraProjectionMatrix);
		_transparentFinalProgram->setUniform2f ("screenSize", _width, _height);
		_transparentFinalProgram->setUniform1f ("step", 1.f);
	} catch (GL::Exception & e) {
		qDebug () << e.msg ().c_str () << endl;
		exit (EXIT_FAILURE);
	}
	glEnable (GL_TEXTURE_2D);
	glActiveTexture (GL_TEXTURE0);
	glBindTexture (GL_TEXTURE_2D, _accumTex);
	glActiveTexture (GL_TEXTURE0 + 1);
	glBindTexture (GL_TEXTURE_2D, _revealTex);
	glActiveTexture (GL_TEXTURE0 + 2);
	glBindTexture (GL_TEXTURE_2D, _opaqueTex);
	glDisable (GL_DEPTH_TEST);
	drawFullScreenQuad ();

	glEnable (GL_BLEND);
	glEnable (GL_DEPTH_TEST);
	glBindFramebuffer (GL_READ_FRAMEBUFFER, 0);
}

void BroxyViewer::resizeGL (int width, int height) {
	_width = width;
	_height = height;
	QGLViewer::resizeGL (width, height);
	initTransparencyBuffers ();
}

// ****************************************************************************
// Quad-Dominant transformation
// ****************************************************************************

void BroxyViewer::updateCageExtras () {
	_cage.recomputeNormals();
	_cage.computeTriNormals (_cageTriNormals);
	_triQuadCageIndices.clear ();
	std::vector<std::vector<unsigned int> > triQuads;
	_cage.triQuadrangulate (triQuads);
	_triQuads = triQuads;
	for (unsigned int i = 0; i < triQuads.size (); i++)
		for (unsigned int j = 0; j < triQuads[i].size (); j++) {
			_triQuadCageIndices.push_back (triQuads[i][j]);
			_triQuadCageIndices.push_back (triQuads[i][(j + 1) % triQuads[i].size ()]);
		}
}

void BroxyViewer::mousePressEvent (QMouseEvent * e) {
	if (e->button () == Qt::MiddleButton && _scaleEditMode != NO_EDITION) {
		double time1, time2;
		time1 = GET_TIME ();
		_scale_field.UpdateGrid ();
		gmorpho.Update (_morpho_mesh, _scale_field, frame_field);
		time2 = GET_TIME ();
		std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
		updateGL ();
	} else if (e->button () == Qt::MiddleButton) {
		const std::vector<Vec3f> & positions = _mesh.P ();
		const std::vector< Vec3<unsigned int> > & triangles = _mesh.T ();
		qglviewer::Vec orig, dir;
		Vec3f origin, direction;
		camera ()->convertClickToLine (e->pos (), orig, dir);
		origin = Vec3f (orig.x, orig.y, orig.z);
		direction = Vec3f (dir.x, dir.y, dir.z);

		Ray ray (origin, direction);
		float minT = std::numeric_limits<float>::infinity();
		for (int i = 0; i < ((int) triangles.size ()); i++) {
			float u, v, t;
			bool isIntersected = false;
			Vec3<unsigned int> tri = triangles[i];
			isIntersected = ray.triangleIntersect (positions[tri[0]],
			                                       positions[tri[1]],
			                                       positions[tri[2]],
			                                       u, v, t);
			if (isIntersected && t < minT) {
				minT = t;
			}
		}

		if (minT != std::numeric_limits<float>::infinity()) {
			if (e->modifiers () && Qt::ControlModifier) {
				Vec3f traced_pos = origin + minT * direction;
				_curr_frame = frame_field.GetHardConstrainedPointId (traced_pos);
				std::cout << "curr_frame : " << _curr_frame << std::endl;
			} else {
				//				Vec3f v0 (1, 1, 0.5);
				Vec3f v0 (1, 0, 0);
				v0 = normalize (v0);
				//				v0 = ((float)(3.f*M_PI/4.f))*v0;
				v0 = ((float)(M_PI / 200.f)) * v0;
				frame_field.AddHardConstrainedPoint (origin + minT * direction, v0);
				bool update_only = false;
				bool use_group_symmetry = true;
				generateFrameField (update_only, use_group_symmetry);
				//				frame_field.GenerateFromSparseBiharmonicSystem ();
				//				frame_field.ComputeOpenGLFrameFieldSliceXY ();
				//				frame_field.ComputeOpenGLHardConstrainedFrames ();
				//				_frames_tri_p = frame_field.opengl_frame_field_p ();
				//				_frames_tri_n = frame_field.opengl_frame_field_n ();
				//				_hc_frames_tri_p = frame_field.opengl_hc_frames_p ();
				//				_hc_frames_tri_n = frame_field.opengl_hc_frames_n ();
				//				updateGL ();
			}
		}
	} else if (e->button () == Qt::LeftButton) {
		_drag_start = Vec3f (e->pos ().x (), e->pos ().y (), 0);
		if (_curr_frame >= 0)
			_curr_twist = frame_field.GetHardConstrainedPointTwist (_curr_frame);
	}
	QGLViewer::mousePressEvent (e);
}

void BroxyViewer::mouseMoveEvent (QMouseEvent * e) {
	if (_scale_field.points ().size () != 0 && _scaleEditMode != NO_EDITION) {
		const std::vector<Vec3f> & positions = _mesh.P ();
		const std::vector< Vec3<unsigned int> > & triangles = _mesh.T ();
		qglviewer::Vec orig, dir;
		Vec3f origin, direction;
		camera ()->convertClickToLine (e->pos (), orig, dir);
		origin = Vec3f (orig.x, orig.y, orig.z);
		direction = Vec3f (dir.x, dir.y, dir.z);

		Ray ray (origin, direction);
		float minT = std::numeric_limits<float>::infinity();
		#pragma omp parallel for
		for (int i = 0; i < ((int) triangles.size ()); i++) {
			float u, v, t;
			bool isIntersected = false;
			Vec3<unsigned int> tri = triangles[i];
			isIntersected = ray.triangleIntersect (positions[tri[0]],
			                                       positions[tri[1]],
			                                       positions[tri[2]],
			                                       u, v, t);
			if (isIntersected) {
				#pragma omp critical
				minT = std::min (t, minT);
			}
		}

		if (minT != std::numeric_limits<float>::infinity()) {
			_scale_field.points ()[0].set_position (origin + minT * direction);
			_drawScaleField = true;
		} else {
			_drawScaleField = false;
		}
		updateGL ();
	}
	if (QApplication::keyboardModifiers ().testFlag (Qt::ControlModifier) == true
	        && QApplication::mouseButtons ().testFlag(Qt::LeftButton) == true) {
		_drag_end = Vec3f (e->pos ().x (), e->pos ().y (), 0);
		Vec3f v = _curr_twist;
		float theta = length (v);
		v = normalize (v);
		float theta_x = (_drag_end[1] - _drag_start[1]) / 256;
		float theta_y = (_drag_end[0] - _drag_start[0]) / 256;
		qglviewer::Vec upv = camera ()->upVector ();
		qglviewer::Vec right_vector = camera ()->rightVector ();

		Vec3f v_x (right_vector.x, right_vector.y, right_vector.z);
		Vec3f v_y (upv.x, upv.y, upv.z);

		v_x = normalize (v_x);
		v_y = normalize (v_y);

		Vec3f e_x (1, 0, 0);
		Vec3f e_y (0, 1, 0);
		Vec3f e_z (0, 0, 1);

		e_x = std::cos (theta) * e_x + std::sin (theta) * cross (v, e_x)
		      + (1.f - std::cos (theta)) * dot (v, e_x) * v;
		e_y = std::cos (theta) * e_y + std::sin (theta) * cross (v, e_y)
		      + (1.f - std::cos (theta)) * dot (v, e_y) * v;
		e_z = std::cos (theta) * e_z + std::sin (theta) * cross (v, e_z)
		      + (1.f - std::cos (theta)) * dot (v, e_z) * v;

		e_x = std::cos (theta_x) * e_x + std::sin (theta_x) * cross (v_x, e_x)
		      + (1.f - std::cos (theta_x)) * dot (v_x, e_x) * v_x;
		e_y = std::cos (theta_x) * e_y + std::sin (theta_x) * cross (v_x, e_y)
		      + (1.f - std::cos (theta_x)) * dot (v_x, e_y) * v_x;
		e_z = std::cos (theta_x) * e_z + std::sin (theta_x) * cross (v_x, e_z)
		      + (1.f - std::cos (theta_x)) * dot (v_x, e_z) * v_x;

		e_x = std::cos (theta_y) * e_x + std::sin (theta_y) * cross (v_y, e_x)
		      + (1.f - std::cos (theta_y)) * dot (v_y, e_x) * v_y;
		e_y = std::cos (theta_y) * e_y + std::sin (theta_y) * cross (v_y, e_y)
		      + (1.f - std::cos (theta_y)) * dot (v_y, e_y) * v_y;
		e_z = std::cos (theta_y) * e_z + std::sin (theta_y) * cross (v_y, e_z)
		      + (1.f - std::cos (theta_y)) * dot (v_y, e_z) * v_y;
		frame_field.UpdateHardConstrainedPoint (_curr_frame, e_x, e_y, e_z);
		frame_field.ComputeOpenGLHardConstrainedFrames ();
		_hc_frames_tri_p = frame_field.opengl_hc_frames_p ();
		_hc_frames_tri_n = frame_field.opengl_hc_frames_n ();
		updateGL ();
		//		bool update_only = true;
		//		generateFrameField (update_only);
	}
	QGLViewer::mouseMoveEvent (e);
}

void BroxyViewer::keyPressEvent (QKeyEvent * e) {
	if  (e->key() == Qt::Key_F6) {
		std::cout << "F6 Pressed" << std::endl;
		updateGL ();
	} else if (e->key () == Qt::Key_A) {
		increaseBaseScale ();
	} else if (e->key () == Qt::Key_Z) {
		decreaseBaseScale ();
	} else if (e->key () == Qt::Key_E || e->key () == Qt::Key_R) {
		if (_scale_field.points ().size () != 0) {
			float scale = _scale_field.points ()[0].scale ();
			if (e->key () == Qt::Key_E)
				scale -= 0.5f * gmorpho.cell_size ();
			else if (e->key () == Qt::Key_R)
				scale += 0.5f * gmorpho.cell_size ();

			scale = std::max (scale,
			                  0.f);
			scale = std::min (scale,
			                  MAX_SE_SIZE);

			if (_scale_field.points ()[0].support () < scale)
				_scale_field.points ()[0].set_support (scale);

			_scale_field.points ()[0].set_scale (scale);
			updateGL ();
		}
	} else if (e->key () == Qt::Key_Space) {
		if ((e->modifiers () && Qt::ControlModifier)) {
			toggleAnimation ();
		} else {
			switch (_scaleEditMode) {
			case (NO_EDITION) :
				_scaleEditMode = MAX_BRUSH;
				break;
			case (MAX_BRUSH) :
				_scaleEditMode = MIN_BRUSH;
				break;
			case (MIN_BRUSH) :
				_scaleEditMode = ZERO_BRUSH;
				break;
			case (ZERO_BRUSH) :
				_scaleEditMode = NO_EDITION;
				break;
			}
			_scale_field.points ()[0].set_edit_mode (_scaleEditMode);
		}
		updateGL ();
	} else if (e->key () == Qt::Key_D || e->key () == Qt::Key_F) {
		if (_scale_field.points ().size () != 0) {
			float support = _scale_field.points ()[0].support ();
			if (e->key () == Qt::Key_D)
				support -= 0.5f * gmorpho.cell_size ();
			else if (e->key () == Qt::Key_F)
				support += 0.5f * gmorpho.cell_size ();

			support = std::max (support,
			                    gmorpho.cell_size ());

			if (_scale_field.points ()[0].scale () > support)
				_scale_field.points ()[0].set_scale (support);

			_scale_field.points ()[0].set_support (support);
			updateGL ();
		}
	} else if (e->key () == Qt::Key_N) {
		double time1, time2;
		if (_bilateralMorphoMeshNormals)
			_bilateralMorphoMeshNormals = false;
		else
			_bilateralMorphoMeshNormals = true;
		gmorpho.set_bilateral_filtering (_bilateralMorphoMeshNormals);
		time1 = GET_TIME ();
		gmorpho.Update (_morpho_mesh, _scale_field, frame_field);
		time2 = GET_TIME ();
		std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
		updateGL ();
	} else if (e->key () == Qt::Key_S) {
		toggleDisplaySmoothShading (!isDisplaySmoothShading ());
	} else if (e->key () == Qt::Key_H) {
		toggleDisplayInputMesh (!isDisplayInputMesh ());
	} else if (e->key () == Qt::Key_J) {
		toggleDisplayProxy (!isDisplayProxy ());
	} else if (e->key () == Qt::Key_K) {
		toggleDisplayProxyMesh (!isDisplayProxyMesh ());
	} else if (e->key () == Qt::Key_Y) {
		toggleDisplayTransparentInputMesh (!isDisplayTransparentInputMesh ());
	} else if (e->key () == Qt::Key_U) {
		toggleDisplayTransparentProxy (!isDisplayTransparentProxy ());
	} else if (e->key () == Qt::Key_I) {
		toggleDisplayTransparentProxyMesh (!isDisplayTransparentProxyMesh ());
	} else if (e->key () == Qt::Key_B) {
		saveSnapshot (false);
		saveStateToFile ();
	} else if (e->key () == Qt::Key_C) {
		if (_drawQuads)
			_drawQuads = false;
		else
			_drawQuads = true;
		updateGL ();
	} else if (e->key () == Qt::Key_L) {
		_isFrameFieldTransparent = _isFrameFieldTransparent ? false : true;
		updateGL ();
	} else if (e->key () == Qt::Key_O) {
		_drawFrameField = _drawFrameField ? false : true;
		updateGL ();
	} else if (e->key () == Qt::Key_M) {
		toggleAsymmetricClosing (!gmorpho.use_asymmetric_closing ());
	} else if (e->key () == Qt::Key_Left || e->key () == Qt::Key_Right) {
		if (e->key () == Qt::Key_Left) {
			frame_field.PreviousSliceXY ();
		}
		if (e->key () == Qt::Key_Right) {
			frame_field.NextSliceXY ();
		}
		frame_field.ComputeOpenGLFrameFieldSliceXY ();
		frame_field.ComputeOpenGLHardConstrainedFrames ();
		_frames_tri_p = frame_field.opengl_frame_field_p ();
		_frames_tri_n = frame_field.opengl_frame_field_n ();
		_hc_frames_tri_p = frame_field.opengl_hc_frames_p ();
		_hc_frames_tri_n = frame_field.opengl_hc_frames_n ();
		updateGL ();
	} else
		QGLViewer::keyPressEvent (e);
}

void BroxyViewer::generateFrameField (bool update_only, bool use_group_symmetry) {
	frame_field.Generate (update_only, use_group_symmetry);
	frame_field.ComputeOpenGLFrameFieldSliceXY ();
	_frames_tri_p = frame_field.opengl_frame_field_p ();
	_frames_tri_n = frame_field.opengl_frame_field_n ();
	frame_field.ComputeOpenGLHardConstrainedFrames ();
	_hc_frames_tri_p = frame_field.opengl_hc_frames_p ();
	_hc_frames_tri_n = frame_field.opengl_hc_frames_n ();
	updateGL ();
}

void BroxyViewer::animate () {
	qglviewer::Camera * old_cam = camera ();
	qglviewer::Camera * new_cam = new qglviewer::Camera (*old_cam);
	qglviewer::Vec scene_center = sceneCenter ();
	qglviewer::Vec new_cam_pos;
	if (ROT_AXIS == 0)
		new_cam_pos = qglviewer::Vec (0,
		                              cos (curr_cam_theta_),
		                              sin (curr_cam_theta_)); // X axis
	else if (ROT_AXIS == 1)
		new_cam_pos = qglviewer::Vec (sin (curr_cam_theta_),
		                              0,
		                              cos (curr_cam_theta_)); // Y axis
	else
		new_cam_pos = qglviewer::Vec (cos (curr_cam_theta_),
		                              sin (curr_cam_theta_),
		                              0); // Z axis
	new_cam_pos = scene_center + DIST_TO_CENTER * new_cam_pos;
	new_cam->setPosition (new_cam_pos);
	new_cam->setUpVector (UP_VECTOR);
	new_cam->lookAt (scene_center);
	setCamera (new_cam);
	curr_cam_theta_ += M_PI / 180.f;
	//	std::cout << curr_cam_theta_ << std::endl;
	delete old_cam;
}

void BroxyViewer::init () {
	// Interface Rendering Intialization
	glewInit();
	if (!GLEW_VERSION_3_0) {
		std::cerr << "Driver does not support OpenGL v3.0" << std::endl;
		exit (EXIT_FAILURE);
	}
	restoreStateFromFile();
	setAutoBufferSwap (true);
	setBackgroundColor (QColor (255, 255, 255));
	glEnable (GL_BLEND);
	glLineWidth (2.f);
	glEnable (GL_LINE_SMOOTH);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Transparency Rendering Intialization
	initTransparencyBuffers ();

	// Screen Shot Initialization
	_takeScreenShot = false;
	setSnapshotFormat ("PNG");

	// Scale Set Initialization
	_scaleEditMode = NO_EDITION;
	_scale_field.AddPoint (Vec3f (0, 0, 0), 0.1f, 0.2f);
	_scalePointMesh.load ("Resources/Models/sphere.off");

	// Set Initial Mesh
	loadMesh ("Resources/Models/beast.off");

	// Intialize Input Interface
	setMouseTracking (true);

	// Shaders Intialization
	try {
		_meshProgram = GL::Program::genVFProgram ("Fixed GGX Shader",
		               "Resources/Shaders/fixed_ggx.vert",
		               "Resources/Shaders/fixed_ggx.frag");
		_scale_fieldProgram = GL::Program::genVFProgram ("Fixed Phong Shader",
		                      "Resources/Shaders/scale_ggx.vert",
		                      "Resources/Shaders/scale_ggx.frag");
		_cageProgram = GL::Program::genVFProgram ("Fixed Phong Shader",
		               "Resources/Shaders/cage.vert",
		               "Resources/Shaders/cage.frag");
		_transparentProgram = GL::Program::genVFProgram ("Fixed Transparency Phong Shader",
		                      "Resources/Shaders/transparent.vert",
		                      "Resources/Shaders/transparent.frag");
		_transparentFinalProgram = GL::Program::genVFProgram ("Fixed Transparency Final Phong Shader",
		                           "Resources/Shaders/transparent_final.vert",
		                           "Resources/Shaders/transparent_final.frag");

	} catch (GL::Exception & e) {
		qDebug () << e.msg ().c_str () << endl;
		exit (EXIT_FAILURE);
	}
}

QString BroxyViewer::helpString() const {
	QString text("<h2>Blade Simple App Demo</h2>");
	text += "<hr><br>";
	text += "Author : Tamy Boubekeur<br>";
	text += "<hr><br>";
	text += "Usage: ./BladeSimpleApp <br>";
	text += "<hr><br>";
	text += "Keyboard commands<br>";
	text += "<hr><br>";
	text += " ?: Print help<br>";
	text += " q, <esc>: Quit<br>";
	text += "<hr><br>";
	return text;
}


void BroxyViewer::featureAwareDecimation (bool on_morpho_output) {
	std::vector<bool> feature_taggs (_morpho_mesh.P ().size (), false);
	ScaleField & scale_field = _scale_field;
	char * scale_grid_cpu = NULL;
	Vec3f bbox_min = scale_field.bbox_min ();
	Vec3f bbox_max = scale_field.bbox_max ();
	float cell_size = scale_field.cell_size ();
	Vec3<unsigned int> res = scale_field.res ();
	scale_field.GetScaleGridCPU (&scale_grid_cpu);
	bool use_linear_constraints = true, use_features = false;
	Decimator * decimator = NULL;
	double time1, time2;
	float target_error;
	int target_num_faces;
	float max_edge_length_alpha;

	if (gmorpho.use_asymmetric_closing ()) {
		target_error = 0.01f;
		target_num_faces = 20;
		max_edge_length_alpha = 2.f;
	} else {
		target_error = 0.001f;
		target_num_faces = 20;
		max_edge_length_alpha = 1.f;
	}

	time1 = GET_TIME ();
	if (on_morpho_output)
		decimator = new Decimator (_morpho_mesh, target_num_faces,
		                           target_error,
		                           use_linear_constraints,
		                           use_features,
		                           max_edge_length_alpha,
		                           bbox_min, bbox_max,
		                           res, cell_size,
		                           scale_grid_cpu,
		                           feature_taggs,
		                           3 // HC iterations
		                          );
	else {
		decimator = new Decimator (_mesh, target_num_faces,
		                           target_error,
		                           use_linear_constraints,
		                           use_features,
		                           max_edge_length_alpha,
		                           bbox_min, bbox_max,
		                           res, cell_size,
		                           scale_grid_cpu,
		                           feature_taggs,
		                           3 // HC iterations
		                          );
	}
	bool heap_not_empty = true;
	_drawQuads = false;
	_drawMorphoMesh = false;
	_drawCageMesh = true;
	_drawInputMesh = true;
	while (heap_not_empty &&
	        decimator->num_faces () > decimator->target_num_faces () &&
	        decimator->error () < decimator->target_error ()
	      ) {
		heap_not_empty = decimator->Optimize ();
		decimator->GetMesh (_cage);
		_drawQuads = false;
		_cage.recomputeNormals();
		_cage.computeTriNormals (_cageTriNormals);
		updateGL ();
	}
	time2 = GET_TIME ();

	decimator->GetMesh (_cage);
	updateCageExtras ();

	delete scale_grid_cpu;
	std::cout << "Feature-Aware Decimator error : "
	          << decimator->error ()
	          << " and number of faces : "
	          << _cage.T ().size ()
	          << " in " << time2 - time1 << " ms."
	          << std::endl;
	updateGL ();
}





