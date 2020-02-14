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

#include <limits>
#include "Vec3.h"

namespace MorphoGraphics {
    class Ray {
    public:
        inline Ray (const Vec3f & origin, const Vec3f & direction) : _origin (origin), _direction (direction) {}
        inline ~Ray () {}
        inline const Vec3f & origin () const { return _origin; }
        inline const Vec3f & direction () const { return _direction; }
        inline bool triangleIntersect (const Vec3f &p0, const Vec3f &p1, const Vec3f &p2,
                                       float & u, float & v, float & t) const {
            Vec3f edge1 = p1 - p0, edge2 = p2 - p0;
            Vec3f pvec = cross(_direction, edge2);
            float det = dot(edge1, pvec);
            if (det == 0.f)
                return false;
            float inv_det = 1.0f / det;
            Vec3f tvec = _origin - p0;
            u = dot(tvec, pvec) * inv_det;
            Vec3f qvec = cross(tvec, edge1);
            v = dot(_direction, qvec) * inv_det;
            t = dot(edge2, qvec) * inv_det;
            if (u < 0.f || u > 1.f)
                return false;
            if (v >= 0.f && u + v <= 1.f)
                return true;
            return false;
        }
        bool boxIntersect (const Vec3f & boxMin, const Vec3f & boxMax,
                           float & nearT, float & farT) const {
            nearT = -std::numeric_limits<float>::infinity();
            farT  =  std::numeric_limits<float>::infinity();
            Vec3f dRcp (1.f/_direction[0], 1.f/_direction[1], 1.f/_direction[2]);
            for (int i=0; i < 3; i++) {
                const float direction = _direction[i];
                const float origin = _origin[i];
                const float minVal = boxMin[i], maxVal = boxMax[i];
                if (direction == 0) {
                    if (origin < minVal || origin > maxVal)
                        return false;
                } else {

                    float t1 = (minVal - origin) * dRcp[i];
                    float t2 = (maxVal - origin) * dRcp[i];
                    if (t1 > t2)
                        std::swap(t1, t2);
                    nearT = std::max(t1, nearT);
                    farT = std::min(t2, farT);
                    if (!(nearT <= farT))
                        return false;
                }
            }
            return true;
        }
    private:
        Vec3f _origin;
        Vec3f _direction;
    };
}
