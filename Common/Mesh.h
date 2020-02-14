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

#pragma once

#define GLEW_STATIC 1
#include <GL/glew.h>

#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
#include <cstdio>
#include <cstdlib>

#include "Vec3.h"
#include "OpenGL.h"

#define MAX_VBO_FACE_SIZE 1000000
#define BUFFER_OFFSET(i) ((void*)(i))

namespace MorphoGraphics {
    class Mesh {
    public:
        class Exception {
        public:
            inline Exception (const std::string & msg) : _msg ("MorphoGraphics Mesh Exception: " + msg) {}
            inline const std::string & msg () const { return _msg; }
        protected:
            std::string _msg;
        };

        /// Edge, with ascending index order.
        struct Edge {
        public:
            inline Edge (unsigned int v0, unsigned int v1) {
                if (v0 < v1) {
                    v[0] = v0;
                    v[1] = v1;
                } else {
                    v[0] = v1;
                    v[1] = v0;
                }
            }
            inline Edge (const Edge & e) { v[0] = e.v[0]; v[1] = e.v[1]; }
            inline ~Edge () {}
            inline Edge & operator= (const Edge & e) { v[0] = e.v[0]; v[1] = e.v[1]; return (*this); }
            inline bool operator== (const Edge & e) { return (v[0] == e.v[0] && v[1] == e.v[1]); }
            inline unsigned int & operator() (unsigned int i) { return v[i]; }
            inline const unsigned int & operator() (unsigned int i) const { return v[i]; }
            inline bool operator< (const Edge & e) const { return (v[0] < e.v[0] || (v[0] == e.v[0] && v[1] < e.v[1])); }
            inline bool contains (unsigned int i) const { return (v[0] == i || v[1] == i); }
            unsigned int v[2];
        };
        typedef std::map<Edge, unsigned int> EdgeIndexMap;
        typedef Vec3<unsigned int> Triangle;
        typedef std::pair<int, int> TriPair;
        typedef std::map<Edge, TriPair> EdgeTriPairMap;
        typedef std::vector<std::vector<unsigned int> > OneRing;
        /// Per-triangle, per-edge neighbors: TriNeighborhood[i][2] is the 2nd edge neighbor index of triangle i. -1 at border.
/*
        typedef std::pair<unsigned int, unsigned int> TrianglePair;
        class Collapsable {
        public:
            inline Collapsable (const Edge & edge, const TrianglePair & trianglePair, float cost=0.f)
                : _edge (edge), _trianglePair (trianglePair), _cost (cost) {}
            inline ~Collapsable () {}
            inline const Edge & edge () const { return _edge; }
            inline const TrianglePair & trianglePair () const { return _trianglePair; }
            inline float cost () const { return _cost; }
            inline bool operator< (const Collapsable & c) const { return (cost () < c.cost ());}
        private:
            Edge _edge;
            TrianglePair _trianglePair;
            float _cost;
        };
        typedef std::priority_queue<Collapsable> CollapsingQueue;
*/
        typedef enum {POINT_RENDERING_MODE=0, WIRE_RENDERING_MODE=1, FLAT_RENDERING_MODE=2, SOLIDWIRE_RENDERING_MODE=3, SMOOTH_RENDERING_MODE=4, QUADS_RENDERING_MODE = 5} RenderingMode;
        inline Mesh ();
        inline virtual ~Mesh ();
        /// Vertex positions
        inline std::vector<Vec3f> & P () { return _P; }
        /// Vertex normals
        inline std::vector<Vec3f> & N () { return _N; }
        inline std::vector<Triangle> & T () { return _T; }
        inline const std::vector<Vec3f> & P () const { return _P; }
        inline const std::vector<Vec3f> & N () const { return _N; }
        inline const std::vector<Triangle> & T () const { return _T; }
        inline void clear ();
        inline void collectOneRing (OneRing & oneRing) const;
        inline void collectEdgeTriPairMap (EdgeTriPairMap & m) const ;
        inline void computeEdgeFaceAdjacency (EdgeIndexMap & edgeMap) const;
        inline void computeDualEdgeMap (EdgeIndexMap & dualVMap1, EdgeIndexMap & dualVMap2) const;
        inline void flipFaceOrientation ();
        inline Vec3f triangleBarycenter (unsigned int i) const;
        inline Vec3f triangleNormal (unsigned int i) const;
        inline void recomputeNormals ();
        inline void smoothNormals ();
        inline void computeTriBarycenters (std::vector<Vec3f> & triBarycenters) const;
        inline void computeTriNormals (std::vector<Vec3f> & triNormals) const;
        inline Vec3f barycenter () const;
        inline float radius () const;
        inline void move (const Vec3f & t);
        inline void scale (float s);
        inline void center ();
        inline void normalize ();
        /// Generate a quad dominant mesh structure by merging pairs of triangles forming acceptable quads w.r.t. the threshold.
        inline void triQuadrangulate (std::vector<std::vector<unsigned int> > & faces, float threshold = 1.0f) const;
        //inline void simplify (unsigned int targetTriCount);
        inline bool useVBO () const;
        inline void toggleUseVBO (bool b);
        inline void initVBO (bool use_vbo = true);
        inline void clearVBO ();
        inline void drawVBO ();
        inline void draw (RenderingMode m = SMOOTH_RENDERING_MODE);
        inline void draw (const std::vector<std::vector<unsigned int> > & faces, RenderingMode m = SMOOTH_RENDERING_MODE);
        inline void load (const std::string & filename);
        inline void store (const std::string & filename);
    private:
        inline bool triContains (const Triangle & t, unsigned int v) const;
		class EdgeEl {
        public:
            unsigned int t0;
            unsigned int t1;
            std::vector<unsigned int> v;
            float qe;
            inline EdgeEl (unsigned int t0, unsigned int t1, std::vector<unsigned int> v, float qe) : t0(t0), t1(t1), v(v), qe(qe) {}
        };
        struct compareEdgeEl {
            inline bool operator() (const EdgeEl x, const EdgeEl y) { return (x.qe > y.qe); }
        };

        inline void loadOFF (const std::string & filename);
        inline float quadError (const Vec3f & v1, const Vec3f & v2, const Vec3f & v3, const Vec3f & v4) const;
        inline void storeOFF (const std::string & filename);
        inline void loadPN (const std::string & filename);
        inline void storePN (const std::string & filename);
        inline void loadOBJ (const std::string & filename);

        std::vector<Vec3f> _P;
        std::vector<Vec3f> _N;
        std::vector<Triangle> _T;
        bool use_vbo_;
        GLuint vertex_vbo_id_;
        GLuint normal_vbo_id_;
        GLuint index_vbo_id_;
    };

    Mesh::Mesh () {
        use_vbo_ = false;
        vertex_vbo_id_ = 0;
        normal_vbo_id_ = 0;
        index_vbo_id_ = 0;
    }

    Mesh::~Mesh () {
    }

    void Mesh::clear () {
        _P.clear ();
        _N.clear ();
        _T.clear ();
        clearVBO ();
    }

    Vec3f Mesh::triangleNormal (unsigned int i) const {
        return MorphoGraphics::normalize (cross (_P[_T[i][1]] - _P[_T[i][0]],
                                        _P[_T[i][2]] - _P[_T[i][0]]));
    }

    Vec3f Mesh::triangleBarycenter (unsigned int i) const {
        return (_P[_T[i][0]] + _P[_T[i][1]] + _P[_T[i][2]])/3.f;
    }

    void Mesh::collectOneRing (OneRing & oneRing) const {
        oneRing.resize (_P.size ());
        for (unsigned int i = 0; i < _T.size (); i++) {
            for (unsigned int j = 0; j < 3; j++) {
                unsigned int vj = _T[i][j];
                for (unsigned int k = 1; k < 3; k++) {
                    unsigned int vk = _T[i][(j+k)%3];
                    if (find (oneRing[vj].begin (), oneRing[vj].end (), vk) == oneRing[vj].end ())
                        oneRing[vj].push_back (vk);
                }
            }
        }
    }

    void Mesh::collectEdgeTriPairMap (Mesh::EdgeTriPairMap & m) const {
        m.clear ();
        for (unsigned int i = 0; i < _T.size (); i++)
            for (unsigned int j = 0; j < 3; j++) {
                Edge e (_T[i][j], _T[i][(j+1)%3]);
                if (m.find (e) != m.end ())
                    m[e].second = i;
                else
                    m[e] = TriPair (i, -1);
            }
    }

    void Mesh::computeEdgeFaceAdjacency (EdgeIndexMap & edgeMap) const{
        for (unsigned int i = 0; i < _T.size (); i++) {
            for (unsigned int j = 0; j < 3; j++) {
                Edge e (_T[i][j], _T[i][(j+1)%3]);
                if (edgeMap.find (e) == edgeMap.end ())
                    edgeMap[e] = 0;
                else
                    edgeMap[e] += 1;
            }
        }
    }

    void Mesh::computeDualEdgeMap (EdgeIndexMap & dualVMap1, EdgeIndexMap & dualVMap2) const{
        for (unsigned int i = 0; i < _T.size (); i++) {
            for (unsigned int j = 0; j < 3; j++) {
                Edge e (_T[i][j], _T[i][(j+1)%3]);
                if (dualVMap1.find (e) == dualVMap1.end ())
                    dualVMap1[e] = _T[i][(j+2)%3];
                else
                    dualVMap2[e] = _T[i][(j+2)%3];
            }
        }
    }

    Vec3f Mesh::barycenter () const {
        if (_P.size () == 0)
            return Vec3f ();
        Vec3f b;
        for (unsigned int i = 0; i < _P.size (); i++)
            b += _P[i];
        return b/_P.size ();
    }

    float Mesh::radius () const {
        float r = 0.f;
        Vec3f b (barycenter ());
        for (unsigned int i = 0; i < _P.size (); i++) {
            float ri = (_P[i] - b).squaredLength ();
            if (ri > r)
                r = ri;
        }
        return sqrt (r);
    }

    void Mesh::move (const Vec3f & t) {
        for (unsigned int i = 0; i < _P.size (); i++)
            _P[i] += t;
    }

    void Mesh::scale (float s) {
        for (unsigned int i = 0; i < _P.size (); i++)
            _P[i] *= s;
    }

    void Mesh::center () {
        move (-barycenter ());
    }

    void Mesh::normalize () {
        scale (1.f/radius ());
    }

    void Mesh::flipFaceOrientation () {
        for (unsigned int i = 0; i < _T.size (); i++) {
            unsigned int tmp = _T[i][1];
            _T[i][1] = _T[i][2];
            _T[i][2] = tmp;
        }
        recomputeNormals ();
    }

    void Mesh::recomputeNormals () {
        _N.resize (_P.size (), Vec3f ());
        for (unsigned int i = 0; i < _T.size (); i++) {
            Vec3f nt (triangleNormal (i));
            for (unsigned int j = 0; j < 3; j++)
                _N[_T[i][j]] += nt;
        }
        for (unsigned int i = 0; i < _N.size (); i++)
            _N[i].normalize ();
    }

    void Mesh::computeTriBarycenters (std::vector<Vec3f> & triBarycenters) const {
        triBarycenters.clear ();
        triBarycenters.resize (_T.size ());
        for (unsigned int i = 0; i < _T.size (); i++)
            triBarycenters[i] = triangleBarycenter (i);
    }

    void Mesh::computeTriNormals (std::vector<Vec3f> & triNormals) const {
        triNormals.clear ();
        triNormals.resize (_T.size ());
        for (unsigned int i = 0; i < _T.size (); i++)
            triNormals[i] = triangleNormal (i);
    }

    

    void Mesh::smoothNormals () {
        std::vector<Vec3f> n (_N.size ());
        for (unsigned int i = 0; i < _T.size (); i++)
            for (unsigned int j = 0; j < 3; j++)
                for (unsigned int k = 1; k < 3; k++)
                    n[_T[i][j]] += _N[_T[i][(j+k)%3]];
        for (unsigned int i = 0; i < _N.size (); i++) {
                n[i].normalize ();
               _N[i] = n[i];
           }
    }

   
 	inline float Mesh::quadError (const Vec3f & v1, const Vec3f & v2, const Vec3f & v3, const Vec3f & v4) const {
     return  (length (cross (v2 - v1 , v3 - v4)) +
              length (cross (v3 - v2 , v4 - v1))) /
             (length (cross (v2 - v1 , v3 - v1)) +
              length (cross (v3 - v4 , v4 - v1)));
    }

    inline bool Mesh::triContains (const Triangle & t, unsigned int v) const {
        return (t[0] == v || t[1] == v || t[2] == v);
    }
    void Mesh::triQuadrangulate (std::vector<std::vector<unsigned int> > & faces, float threshold) const {
        faces.clear ();
        std::vector<Vec3f> N;
        computeTriNormals (N);
        Mesh::EdgeTriPairMap M;
        collectEdgeTriPairMap (M);
        std::vector<bool> marked (_T.size (), false);
        std::priority_queue<EdgeEl, std::vector<EdgeEl>, compareEdgeEl> PQ;
        for (Mesh::EdgeTriPairMap::iterator it = M.begin (); it != M.end (); it++)
            if (it->second.second != -1) {
                unsigned int t0 = static_cast<unsigned int>(it->second.first);
                unsigned int t1 = static_cast<unsigned int>(it->second.second);
                std::vector<unsigned int> p (4);
                unsigned int e = 0;
                for (unsigned int i = 0; i < 3; i++)
                    if (Edge (_T[t0][i], _T[t0][(i+1)%3]) == it->first)
                        e = i;
                unsigned int q = 0;
                for (unsigned int i = 0; i < 3; i++)
                    if (Edge (_T[t1][i], _T[t1][(i+1)%3]) == it->first)
                        q = _T[t1][(i+2)%3];
                if (e == 0) {
                    p[0] = _T[t0][0]; p[1] = q; p[2] = _T[t0][1]; p[3] = _T[t0][2];
                } else if (e == 1) {
                    p[0] = _T[t0][0]; p[1] = _T[t0][1]; p[2] = q; p[3] = _T[t0][2];
                } else {
                    p[0] = _T[t0][0]; p[1] = _T[t0][1]; p[2] = _T[t0][2]; p[3] = q;
                }
                float qe = quadError (_P[p[0]], _P[p[1]], _P[p[2]], _P[p[3]]);
                PQ.push (EdgeEl (t0, t1, p, qe));
            }
        while (!PQ.empty ()) {
            EdgeEl el = PQ.top ();
            PQ.pop ();
            if (!marked[el.t0] && !marked[el.t1] && el.qe < threshold) {
                faces.push_back (el.v);
                marked[el.t0] = true;
                marked[el.t1] = true;
            }
        }
        for (int i = 0; i < int (_T.size ()); i++)
            if (!marked[i]) {
                std::vector<unsigned int> f (3);
                for (unsigned int j = 0; j < 3; j++)
                    f[j] = _T[i][j];
                faces.push_back (f);
            }

    }

    bool Mesh::useVBO () const {
        return use_vbo_;
    }

    void Mesh::toggleUseVBO (bool b) {
        use_vbo_ = b;
    }

    void Mesh::initVBO (bool use_vbo) {
        // First clear the VBO
        clearVBO ();

        // By default use_vbo_ set to 'true' since we are constructing one
        use_vbo_ = use_vbo;

        glGenBuffers (1, &vertex_vbo_id_);
        glBindBuffer (GL_ARRAY_BUFFER, vertex_vbo_id_);
        MorphoGraphics::GL::printOpenGLError ("Binding Vertex VBO");
        glBufferData (GL_ARRAY_BUFFER, sizeof (Vec3f)*_P.size (), (GLvoid*)(_P.data ()), GL_STATIC_DRAW);
        MorphoGraphics::GL::printOpenGLError ("Allocating Vertex VBO");

        glGenBuffers (1, &normal_vbo_id_);
        glBindBuffer (GL_ARRAY_BUFFER, normal_vbo_id_);
        MorphoGraphics::GL::printOpenGLError ("Binding Normal VBO");
        glBufferData (GL_ARRAY_BUFFER, sizeof (Vec3f)*_N.size (), (GLvoid*)(_N.data ()), GL_STATIC_DRAW);
        MorphoGraphics::GL::printOpenGLError ("Allocating Normal VBO");

        glGenBuffers (1, &index_vbo_id_);
        glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, index_vbo_id_);
        MorphoGraphics::GL::printOpenGLError ("Binding Indices VBO");
        glBufferData (GL_ELEMENT_ARRAY_BUFFER, sizeof (Triangle)*_T.size (), (GLvoid*)(_T.data()), GL_STATIC_DRAW);
        MorphoGraphics::GL::printOpenGLError ("Allocating Indices VBO");

        glBindBuffer (GL_ARRAY_BUFFER, vertex_vbo_id_);
        glEnableClientState (GL_VERTEX_ARRAY);
        glVertexPointer (3, GL_FLOAT, sizeof (Vec3f), BUFFER_OFFSET(0));   //The starting point of the VBO, for the vertices
        MorphoGraphics::GL::printOpenGLError ("Setting Vertex pointer");

        glBindBuffer (GL_ARRAY_BUFFER, normal_vbo_id_);
        glEnableClientState (GL_NORMAL_ARRAY);
        glNormalPointer (GL_FLOAT, sizeof (Vec3f), BUFFER_OFFSET(0));   //The starting point of normals, 12 bytes away
        MorphoGraphics::GL::printOpenGLError ("Setting Normal pointer");

        glBindBuffer (GL_ARRAY_BUFFER, 0);
        MorphoGraphics::GL::printOpenGLError ("Binding ARRAY_BUFFER to 0");
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        MorphoGraphics::GL::printOpenGLError ("Binding ELEMENT_ARRAY_BUFFER to 0");
    }

    void Mesh::clearVBO () {
        if (vertex_vbo_id_ != 0)
            glDeleteBuffers (1, &vertex_vbo_id_);
        if (normal_vbo_id_ != 0)
            glDeleteBuffers (1, &normal_vbo_id_);
        if (index_vbo_id_ != 0)
            glDeleteBuffers (1, &index_vbo_id_);
        vertex_vbo_id_ = 0;
        normal_vbo_id_ = 0;
        index_vbo_id_ = 0;
        use_vbo_ = false;
    }

    void Mesh::drawVBO () {
        if (!use_vbo_) {
            glEnableClientState (GL_VERTEX_ARRAY);
            glEnableClientState (GL_NORMAL_ARRAY);
            glVertexPointer (3, GL_FLOAT, sizeof (Vec3f), (GLvoid*)(_P.data ()));
            glNormalPointer (GL_FLOAT, sizeof (Vec3f), (GLvoid*)(_N.data()));
            glDrawElements (GL_TRIANGLES, 3 * _T.size(), GL_UNSIGNED_INT, (GLvoid*)(_T.data()));
            glDisableClientState (GL_VERTEX_ARRAY);
            glDisableClientState (GL_NORMAL_ARRAY);
        } else {
            glBindBuffer (GL_ARRAY_BUFFER, vertex_vbo_id_);
            glVertexPointer (3, GL_FLOAT, sizeof (Vec3f), BUFFER_OFFSET(0));
            MorphoGraphics::GL::printOpenGLError ("Binding vertices VBO before drawing");
            glBindBuffer (GL_ARRAY_BUFFER, normal_vbo_id_);
            glNormalPointer (GL_FLOAT, sizeof (Vec3f), BUFFER_OFFSET(0));
            MorphoGraphics::GL::printOpenGLError ("Binding vertices VBO before drawing");
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_vbo_id_);
            MorphoGraphics::GL::printOpenGLError ("Binding indices VBO before drawing");
            glDrawElements (GL_TRIANGLES, 3 * _T.size(),
                            GL_UNSIGNED_INT, BUFFER_OFFSET(0));
            MorphoGraphics::GL::printOpenGLError ("Drawing the geometry");
            glBindBuffer (GL_ARRAY_BUFFER, 0);
            MorphoGraphics::GL::printOpenGLError ("Unbinding Vertex VBO");
            glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);
            MorphoGraphics::GL::printOpenGLError ("Unbinding Indices VBO");
        }
    }

    inline void glNormalVec3f (const Vec3f & n) {
        glNormal3f (n[0], n[1], n[2]);
    }

    inline void glVertexVec3f (const Vec3f & p) {
        glVertex3f (p[0], p[1], p[2]);
    }

    void Mesh::draw (RenderingMode m) {
        glEnableVertexAttribArray (0);
        glEnableVertexAttribArray (1);
        glPolygonMode (GL_FRONT_AND_BACK, m == WIRE_RENDERING_MODE ? GL_LINE : GL_FILL);
        if (m == POINT_RENDERING_MODE || m == WIRE_RENDERING_MODE || m == SMOOTH_RENDERING_MODE) {
		  /*if (use_vbo_)
                _meshBuffer.draw();
				else {*/
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(&_P[0]));
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(((float*)&_N[0])));
                if (m == POINT_RENDERING_MODE)
                    glDrawArrays (GL_POINTS, 0, _P.size ());
                else
                    glDrawElements (GL_TRIANGLES, 3*_T.size(), GL_UNSIGNED_INT, (GLvoid*)(&_T[0]));

				/*}*/
        } else { // FLAT_RENDERING_MODE and SOLIDWIRE_RENDERING_MODE, per-triangle normal
            std::vector<Vec3f> p (3*_T.size ());
            std::vector<Vec3f> n (3*_T.size ());
            for (unsigned int i = 0; i < _T.size (); i++) {
                Vec3f tn (triangleNormal (i));
                for (unsigned j = 0; j < 3; j++) {
                    unsigned int index = 3*i+j;
                    p[index] = _P[_T[i][j]];
                    n[index] = tn;
                }
            }
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(&p[0]));
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(((float*)&n[0])));
            glDrawArrays (GL_TRIANGLES, 0, p.size ());
        }
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
    }

		void Mesh::draw (const std::vector<std::vector<unsigned int> > & faces, RenderingMode m) {
      	glEnableVertexAttribArray (0);
        glEnableVertexAttribArray (1);
        glPolygonMode (GL_FRONT_AND_BACK, m == WIRE_RENDERING_MODE ? GL_LINE : GL_FILL);
        std::vector<Vec3f> p (3*_T.size ());
        std::vector<Vec3f> n (3*_T.size ());
				int triIndex = 0;
        for (unsigned int i = 0; i < faces.size (); i++) {
					std::vector<unsigned int> face = faces[i];
					Vec3f tn;
					if (face.size () == 3) {
						tn = MorphoGraphics::normalize (cross (_P[face[1]] - _P[face[0]],
                                  		_P[face[2]] - _P[face[0]]));
						for (unsigned j = 0; j < 3; j++) {
            	unsigned int index = 3*triIndex+j;
            	p[index] = _P[face[j]];
            	n[index] = tn;
          	}
						triIndex++;
					} else if (face.size () == 4) {
						tn = MorphoGraphics::normalize (cross (_P[face[1]] - _P[face[0]],
                                  		_P[face[2]] - _P[face[0]]));
						tn = tn + MorphoGraphics::normalize (cross (_P[face[2]] - _P[face[0]],
                                  		_P[face[3]] - _P[face[0]]));
						tn.normalize ();
						for (unsigned j = 0; j < 3; j++) {
            	unsigned int index = 3*triIndex+j;
            	p[index] = _P[face[j]];
            	n[index] = tn;
          	}
						triIndex++;

						for (unsigned j = 0; j < 3; j++) {
            	unsigned int index = 3*triIndex+j;
							if (j == 2)
            		p[index] = _P[face[0]];
							else
            		p[index] = _P[face[(j+2)]];
            	n[index] = tn;
          	}
						triIndex++;
					}
        }
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(&p[0]));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (Vec3f), (GLvoid*)(((float*)&n[0])));
        glDrawArrays (GL_TRIANGLES, 0, p.size ());
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
    }

    void Mesh::load (const std::string & filename) {
        std::string formatString = filename.substr (filename.size ()-3, 3);
        if (formatString == "off")
            loadOFF (filename);
        else if (formatString == ".pn")
            loadPN (filename);
        else if (formatString == "obj")
            loadOBJ (filename);
        else
            throw MorphoGraphics::Mesh::Exception ("Error loading mesh: unrecognized file format.");
    }

    void Mesh::store (const std::string & filename) {
        std::string formatString = filename.substr (filename.size ()-3, 3);
        if (formatString == "off")
            storeOFF (filename);
        else if (formatString == ".pn")
            storePN (filename);
        else
            throw MorphoGraphics::Mesh::Exception ("Error loading mesh: unrecognized file format.");
    }

    void Mesh::loadOFF (const std::string & filename) {
        clear ();
        std::ifstream in (filename.c_str ());
        if (!in)
            throw MorphoGraphics::Mesh::Exception ("Error loading OFF file: " + filename);
        std::string offString;
        unsigned int numV, numF, tmp;
        in >> offString >> numV >> numF >> tmp;
        _P.resize (numV);
        for (unsigned int i = 0; i < numV; i++)
            in >> _P[i];
        unsigned int s;
        for (unsigned int i = 0; i < numF; i++) {
            in >> s;
            std::vector<unsigned int> v(s);
            for (unsigned int j = 0; j < s; j++)
                        in >> v[j];
            for (unsigned int j = 2; j < s; j++)
                _T.push_back (Triangle (v[0], v[j-1], v[j]));
        }
        in.close ();
        recomputeNormals();
    }

    void Mesh::storeOFF (const std::string & filename) {
        std::ofstream out (filename.c_str ());
        if (!out)
            throw MorphoGraphics::Mesh::Exception ("Error storing OFF file: " + filename);
        out << "OFF" << std::endl << _P.size () << " " << _T.size () << " 0" << std::endl;
        for (unsigned int i = 0; i < _P.size (); i++)
            out << _P[i] << std::endl;
        for (unsigned int i = 0; i < _T.size (); i++)
            out << "3 " << _T[i] << std::endl;
        out.close ();
    }

    void Mesh::loadPN (const std::string & filename) {
        clear ();
        FILE * file = fopen (filename.c_str (), "r");
        if (!file)
            throw Exception ("Error loading a PN file: Cannot read file" + std::string (filename));
        fseek (file, 0, SEEK_END);
        unsigned int numOfBytes = ftell (file);
        unsigned int numOfFloats = numOfBytes/sizeof (float);
        unsigned int numOfPts = numOfFloats/6;
        fseek (file, 0, SEEK_SET);
        float * pn = new float[numOfFloats];
				size_t size_read = fread (pn, sizeof (float), numOfFloats, file);
        if (size_read != (numOfFloats*sizeof (float)))
            throw Exception ("Error loading a PN file: Cannot read file" + std::string (filename));
        _P.resize (numOfPts);
        _N.resize (numOfPts);
        for (unsigned int i = 0; i < numOfPts; i++) {
            _P[i] = Vec3f (pn[6*i], pn[6*i+1], pn[6*i+2]);
            _N[i] = Vec3f (pn[6*i+3], pn[6*i+4], pn[6*i+5]);
        }
        delete [] pn;
        fclose (file);
    }

    void Mesh::storePN (const std::string & filename) {
        FILE * file = fopen (filename.c_str (), "w");
        if (!file)
            throw Exception ("Error storing a PN file: Cannot open file" + std::string (filename));
        for (unsigned int i = 0; i < _P.size (); i++) {
            float buf[6];
            buf[0] = _P[i][0]; buf[1] = _P[i][1]; buf[2] = _P[i][2];
            buf[3] = _N[i][0]; buf[4] = _N[i][1]; buf[5] = _N[i][2];
            fwrite (&buf, sizeof (float), 6, file);
        }
        fclose (file);
    }

    void Mesh::loadOBJ (const std::string & filename) {
        clear ();
        std::ifstream in (filename.c_str ());
        if (!in)
            throw Exception ("Error loading OBJ file: " + filename);
        bool stillComments = true;
        do {
            char c = in.get ();
            if (c == '#') {
                char tmp[256];
                in.getline (tmp, 256, '\n');
            } else {
                in.putback (c);
                stillComments = false;
            }
        } while (stillComments);
        std::string mtlFilename;
        std::string buf;
        in >> buf;
        while (!in.eof ()) {
            if (buf == "mtllib") {
                in >> mtlFilename;
                std::ifstream in (filename.c_str ());
                if (!in)
                    throw Exception ("Error loading OBJ file: " + filename);
            } else if (buf == "o") {
                std::string objName;
                in >> objName;
            } else if (buf == "v") {
                Vec3f p;
                in >> p;
                _P.push_back (p);
            } else if (buf == "usemtl") {
                ;
            } else if (buf == "s") {
                unsigned int sNum;
                in >> sNum;
            } else if (buf == "f") {
                std::vector<unsigned int> f;
                while (in.peek () != '\n') {
                    unsigned int v;
                    in >> v;
                    f.push_back (v);
                }
                for (unsigned int i = 2; i < f.size (); i++)
                    _T.push_back(Triangle (f[0]-1, f[i-1]-1, f[i]-1));
            } else {
                throw Exception ("Error loading OBJ File, following token unknown: " + buf);
            }
            in >> buf;
        }
        in.close ();
        recomputeNormals();
    }
}

// Some Emacs-Hints -- please don't remove:
//
//  Local Variables:
//  mode:C++
//  tab-width:4
//  End:

