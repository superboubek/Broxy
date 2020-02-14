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

#include <cmath>
#include <iostream>
#include <algorithm>
#include "Vec3.h"

namespace MorphoGraphics {

    /**
 * A colomn-major 4x4 transformation matrix.
 */
    template <class T>
            class Mat4 {
            public:
        class Exception {
        public:
            inline Exception (const std::string & msg) : _msg ("MorphoGraphics Mat4 Exception: " + msg) {}
            inline const std::string & msg () const { return _msg; }
        protected:
            std::string _msg;
        };

        /// Set to identity by default.
        inline Mat4 (void)	{ loadIdentity (); }
        inline Mat4 (const Mat4 & mat) {
            for (unsigned int i = 0; i < 16; i++)
                m[i] = mat[i];
        }
        ~Mat4() {}
        inline Mat4 (const T * mm) { set (mm); }
        inline Mat4 (T a00, T a01, T a02, T a03,
                     T a10, T a11, T a12, T a13,
                     T a20, T a21, T a22, T a23,
                     T a30, T a31, T a32, T a33) {
            set (a00, a01, a02, a03,
                 a10, a11, a12, a13,
                 a20, a21, a22, a23,
                 a30, a31, a32, a33);
        }
        inline void set (T a00, T a01, T a02, T a03,
                         T a10, T a11, T a12, T a13,
                         T a20, T a21, T a22, T a23,
                         T a30, T a31, T a32, T a33) {
            m[0] = a00; m[1] = a01; m[2] = a02; m[3] = a03;
            m[4] = a10; m[5] = a11; m[6] = a12; m[7] = a13;
            m[8] = a20; m[9] = a21; m[10] = a22; m[11] = a23;
            m[12] = a30; m[13] = a31; m[14] = a32; m[15] = a33;
        }
        inline void set (const T * a) { for (unsigned int i = 0; i < 16; i++) m[i] = a[i]; }
        inline T& operator[] (int index) { return (m[index]); }
        inline const T& operator[] (int index) const { return (m[index]); }
        inline T& operator() (int i, int j) { return (m[4*i+j]); }
        inline const T& operator() (int i, int j) const { return (m[4*i+j]); }
        inline Mat4& operator= (const Mat4 & mat) {
            for (unsigned int i = 0; i < 16; i++)
                m[i] = mat[i];
            return (*this);
        }
        inline Mat4& operator+= (const Mat4 & mat) {
            for (unsigned int i = 0; i < 16; i++)
                m[i] += mat[i];
            return (*this);
        }
        inline Mat4& operator-= (const Mat4 & mat) {
            for (unsigned int i = 0; i < 16; i++)
                m[i] -= mat[i];
            return (*this);
        }
        inline Mat4& operator*= (T & s) {
            for (unsigned int i = 0; i < 16; i++)
                m[i] *= s;
            return (*this);
        }
        inline Mat4& operator*= (const Mat4 & mat) {
            Mat4 tmp;
            for (unsigned int i = 0; i < 4; i++)
                for (unsigned int j = 0; j < 4; j++)
                    for (unsigned int k = 0; k < 4; k++)
                        tmp(i, j) += (*this)(k, j)*mat(j,k);
            (*this) = tmp;
            return (*this);
        }
        inline Mat4 operator+ (const Mat4 & mat) const {
            Mat4 res;
            for (unsigned int i = 0; i < 16; i++)
                res[i] = m[i] + mat[i];
            return res;
        }
        inline Mat4 operator- (const Mat4 & mat) const {
            Mat4 res;
            for (unsigned int i = 0; i < 16; i++)
                res[i] = m[i] - mat[i];
            return res;
        }
        inline Mat4 operator- () const {
            Mat4 res;
            for (unsigned int i = 0; i < 16; i++)
                res[i] = -m[i];
            return res;
        }
        inline Mat4 operator* (const Mat4 & mat) const {
            Mat4 tmp;
            for (unsigned int i = 0; i < 4; i++)
                for (unsigned int j = 0; j < 4; j++)
                    for (unsigned int k = 0; k < 4; k++)
                        tmp(i, j) += (*this)(k, j)*mat(j,k);
            return tmp;
        }
        inline Mat4 operator* (T s) const {
            Mat4 res;
            for (unsigned int i = 0; i < 16; i++)
                res[i] = s*m[i];
            return res;
        }
        inline Vec3<T> operator* (const Vec3<T> & v) const {
            Vec3<T> res;
            for (unsigned int i = 0; i < 3; i++)
                for (unsigned int j = 0; j < 4; j++)
                    res[i] += v[i]*(*this)(j, i);
            return res;
        }
        inline bool operator == (const Mat4 & mat) const {
            for (unsigned int i = 0; i < 16; i++)
                if (m[i] != mat.m[i])
                    return false;
            return true;
        }
        inline bool operator != (const Mat4 & mat) const {
            return !((*this)==mat);
        }
        inline T * data () { return m; }
        inline const T * data () const { return m; }
        inline Mat4 & transpose () {
            for (unsigned int i = 0; i < 4; i++)
                for (unsigned int j = i+1; j < 4; j++)
				  std::swap ((*this) (i, j), (*this) (j, i));
            return (*this);
        }
        inline Mat4 & invert () {
            double inv[16], det;
            int i;
            inv[0] = m[5]  * m[10] * m[15] -
                     m[5]  * m[11] * m[14] -
                     m[9]  * m[6]  * m[15] +
                     m[9]  * m[7]  * m[14] +
                     m[13] * m[6]  * m[11] -
                     m[13] * m[7]  * m[10];

            inv[4] = -m[4]  * m[10] * m[15] +
                     m[4]  * m[11] * m[14] +
                     m[8]  * m[6]  * m[15] -
                     m[8]  * m[7]  * m[14] -
                     m[12] * m[6]  * m[11] +
                     m[12] * m[7]  * m[10];

            inv[8] = m[4]  * m[9] * m[15] -
                     m[4]  * m[11] * m[13] -
                     m[8]  * m[5] * m[15] +
                     m[8]  * m[7] * m[13] +
                     m[12] * m[5] * m[11] -
                     m[12] * m[7] * m[9];

            inv[12] = -m[4]  * m[9] * m[14] +
                      m[4]  * m[10] * m[13] +
                      m[8]  * m[5] * m[14] -
                      m[8]  * m[6] * m[13] -
                      m[12] * m[5] * m[10] +
                      m[12] * m[6] * m[9];

            inv[1] = -m[1]  * m[10] * m[15] +
                     m[1]  * m[11] * m[14] +
                     m[9]  * m[2] * m[15] -
                     m[9]  * m[3] * m[14] -
                     m[13] * m[2] * m[11] +
                     m[13] * m[3] * m[10];

            inv[5] = m[0]  * m[10] * m[15] -
                     m[0]  * m[11] * m[14] -
                     m[8]  * m[2] * m[15] +
                     m[8]  * m[3] * m[14] +
                     m[12] * m[2] * m[11] -
                     m[12] * m[3] * m[10];

            inv[9] = -m[0]  * m[9] * m[15] +
                     m[0]  * m[11] * m[13] +
                     m[8]  * m[1] * m[15] -
                     m[8]  * m[3] * m[13] -
                     m[12] * m[1] * m[11] +
                     m[12] * m[3] * m[9];

            inv[13] = m[0]  * m[9] * m[14] -
                      m[0]  * m[10] * m[13] -
                      m[8]  * m[1] * m[14] +
                      m[8]  * m[2] * m[13] +
                      m[12] * m[1] * m[10] -
                      m[12] * m[2] * m[9];

            inv[2] = m[1]  * m[6] * m[15] -
                     m[1]  * m[7] * m[14] -
                     m[5]  * m[2] * m[15] +
                     m[5]  * m[3] * m[14] +
                     m[13] * m[2] * m[7] -
                     m[13] * m[3] * m[6];

            inv[6] = -m[0]  * m[6] * m[15] +
                     m[0]  * m[7] * m[14] +
                     m[4]  * m[2] * m[15] -
                     m[4]  * m[3] * m[14] -
                     m[12] * m[2] * m[7] +
                     m[12] * m[3] * m[6];

            inv[10] = m[0]  * m[5] * m[15] -
                      m[0]  * m[7] * m[13] -
                      m[4]  * m[1] * m[15] +
                      m[4]  * m[3] * m[13] +
                      m[12] * m[1] * m[7] -
                      m[12] * m[3] * m[5];

            inv[14] = -m[0]  * m[5] * m[14] +
                      m[0]  * m[6] * m[13] +
                      m[4]  * m[1] * m[14] -
                      m[4]  * m[2] * m[13] -
                      m[12] * m[1] * m[6] +
                      m[12] * m[2] * m[5];

            inv[3] = -m[1] * m[6] * m[11] +
                     m[1] * m[7] * m[10] +
                     m[5] * m[2] * m[11] -
                     m[5] * m[3] * m[10] -
                     m[9] * m[2] * m[7] +
                     m[9] * m[3] * m[6];

            inv[7] = m[0] * m[6] * m[11] -
                     m[0] * m[7] * m[10] -
                     m[4] * m[2] * m[11] +
                     m[4] * m[3] * m[10] +
                     m[8] * m[2] * m[7] -
                     m[8] * m[3] * m[6];

            inv[11] = -m[0] * m[5] * m[11] +
                      m[0] * m[7] * m[9] +
                      m[4] * m[1] * m[11] -
                      m[4] * m[3] * m[9] -
                      m[8] * m[1] * m[7] +
                      m[8] * m[3] * m[5];

            inv[15] = m[0] * m[5] * m[10] -
                      m[0] * m[6] * m[9] -
                      m[4] * m[1] * m[10] +
                      m[4] * m[2] * m[9] +
                      m[8] * m[1] * m[6] -
                      m[8] * m[2] * m[5];

            det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
            if (det == 0)
                throw Exception ("Matrix non-invertible (null determinant).");
            det = 1.0 / det;
            for (i = 0; i < 16; i++)
                m[i] = inv[i] * det;
            return (*this);
        }

        inline void setNull () {  set (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
        inline void loadIdentity () { setNull (); m[0] = m[5] = m[10] = m[15] = 1.0; }
        /// Left axis
        inline void setAxis (unsigned int axis, const Vec3<T> & x) { for (unsigned int i = 0; i < 3; i++) (*this)(axis, i) = x[i]; }
        inline Vec3<T> getAxis (unsigned int axis) const { return Vec3<T> (m[4*axis], m[4*axis+1], m[4*axis+2]); }
        inline Vec3<T> getTranslation () const { return Vec3<T> (m[12], m[13], m[14]); }
        inline static Mat4 lookAt(const Vec3<T> & eye, const Vec3<T> & center, const Vec3<T> & up) {
                    Mat4 m;
                    m.loadIdentity();
                    Vec3<T> f (normalize (center - eye));
                    Vec3<T> s = normalize (cross (f, up));
                    Vec3<T> u = normalize (cross (s, f));
                    m[0] = s[0];
                    m[4] = s[1];
                    m[8] = s[2];
                    m[1] = u[0];
                    m[5] = u[1];
                    m[9] = u[2];
                    m[2] = -f[0];
                    m[6] = -f[1];
                    m[10] = -f[2];
                    m[12] = -dot(s, eye);
                    m[13] = -dot(u, eye);
                    m[14] = dot(f, eye);
                    return m;
        }
        inline static Mat4 perspective (float fovy, float aspectRatio, float n, float f) {
            Mat4 m;
            const float deg2rad = M_PI/180.0;
            float fovyRad = deg2rad*fovy;
            float tanHalfFovy = tan (fovyRad/2.0);
            for (unsigned int i = 0; i < 16; i++)
                m[i] = 0.0;
            m[0] = 1.0 / (aspectRatio * tanHalfFovy);
            m[5] = 1.0 / tanHalfFovy;
            m[10] = -(f + n) / (f - n);
            m[11] = -1.0;
            m[14] = -2.0*(f * n) / (f - n);
            return m;
        }
    protected:
        T m[16];
    };

    template <class T> Mat4<T> operator * (const T &s, const Mat4<T> &M) {
        return (M * s);
    }

    template <class T> Mat4<T> transpose (Mat4<T> & M) {
        return M.transpose ();
    }

    template <class T> Mat4<T> inverse (Mat4<T> & M) {
        return M.invert ();
    }

    template <class T> std::ostream & operator<< (std::ostream & output, const Mat4<T> & v) {
        for (unsigned int i = 0; i < 16; i++)
            output << v[0] << (i==15 ? "" : " ");
        return output;
    }

    template <class T> std::istream & operator>> (std::istream & input, Mat4<T> & v) {
        for (unsigned int i = 0; i < 16; i++)
            input >> v[i];
        return input;
    }

    typedef Mat4<float> Mat4f;
    typedef Mat4<double> Mat4d;
    
}
// Some Emacs-Hints -- please don't remove:
//
//  Local Variables:
//  mode:C++
//  tab-width:4
//  End:
