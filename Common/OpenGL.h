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

#ifndef BLADESDK_GL_MODULE
#define BLADESDK_GL_MODULE

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <GL/glew.h>

namespace MorphoGraphics {
	namespace GL {
		static const std::string SHADER_PATH ("Resources/Shaders/");
		class Exception {
			public:
				inline Exception (const std::string & msg) : _msg ("MorphoGraphics GL Exception: " + msg) {}
				inline const std::string & msg () const { return _msg; }
			protected:
				std::string _msg;
		};

		/// Throws an expection if an error occurred.
		static inline void checkGLExceptions ();

		class Shader {
			public:
				inline Shader (const std::string & name, GLuint _type);
				inline virtual ~Shader ();
				inline GLuint id () const { return _id; }
				inline const std::string & name () const { return _name; }
				inline GLenum type () const { return _type; }
				inline const std::string & source () const { return _source; }
				inline const std::string & filename () const { return _filename; }
				inline void setSource (const std::string & source);
				inline void compile ();
				inline void loadFromFile (const std::string & filename);
				inline void reload ();
			protected:
				inline std::string infoLog ();
			private:
				GLuint _id;
				std::string _name;
				GLuint _type;
				std::string _filename;
				std::string _source;
		};

		class Program {
			public:
				inline Program (const std::string & name);
				inline virtual ~Program ();
				inline GLuint id () const { return _id; }
				inline std::string name () const { return _name; }
				inline void attach (Shader * shader);
				inline void detach (Shader * shader);
				inline void link ();
				inline void use ();
				static void stop ();
				inline GLint getUniformLocation (const std::string & uniformName);
				inline void setUniform1f (GLint location, float value);
				inline void setUniform1f (const std::string & name, float value);
				inline void setUniform2f (GLint location, float value0, float value1);
				inline void setUniform2f (const std::string & name, float value0, float value1);
				inline void setUniform3f (GLint location, float value0, float value1, float value2);
				inline void setUniform3f (const std::string & name, float value0, float value1, float vlaue2);
				inline void setUniform4f (GLint location, float value0, float value1, float value2, float value3);
				inline void setUniform4f (const std::string & name, float value0, float value1, float value2, float value3);
				inline void setUniformMatrix4fv (GLint location, const float * values);
				inline void setUniformMatrix4fv (const std::string & name, const float * values);
				inline void setUniformNf (GLint location, unsigned int numValues, const float * values);
				inline void setUniformNf (const std::string & name, unsigned int numValues, const float * values);
				inline void setUniform1i (GLint location, int value);
				inline void setUniform1i (const std::string & name, int value);
				inline void setUniformNi (GLint location, unsigned int numValues, const int * values);
				inline void setUniformNi (const std::string & name, unsigned int numValues, const int * values);
				inline void reload ();
				/// generate a simple program, with only vertex and fragment shaders.
				static Program * genVFProgram (const std::string & name,
																			 const std::string & vertexShaderFilename,
																			 const std::string & fragmentShaderFilename);
				static Program * genVGFProgram (const std::string & name,
																				const std::string & vertexShaderFilename,
																				const std::string & geometryShaderFilename,
																				const std::string & fragmentShaderFilename);
			protected:
				std::string infoLog ();
			private:
				GLuint _id;
				std::string _name;
				std::vector<Shader*>_shaders;
		};

		class Framebuffer {
			public:
				inline Framebuffer ();
				inline virtual ~Framebuffer ();
				inline void bind (GLenum target);
			private:
				GLuint _id;
		};

		class Buffer {
			public:
				inline Buffer ();
				inline virtual ~Buffer ();

				inline GLuint id () const { return _id; }
				inline GLenum target () const { return _target; }
				inline void setTarget (GLenum t) { _target = t; }
				inline GLsizei size () const { return _size; }
				inline GLenum usage () const { return _usage; }
				inline void setUsage (GLenum u) { _usage = u; }
				inline GLenum mode () const { return _mode; }
				inline void setMode (GLenum m) { _mode = m; }

				inline void bind ();
				inline void unbind ();
				inline void setData (GLsizei size, const GLvoid * data);
				inline void clear ();

			private:
				GLuint _id;
				GLenum _target;
				GLsizei _size;
				GLenum _usage;
				GLenum _mode;
		};

		class VertexBuffer : public Buffer {
			public:
				inline VertexBuffer ();
				inline virtual ~VertexBuffer ();
				inline void preDraw ();
				inline void draw ();
				inline void postDraw ();
		};

		class IndexBuffer : public Buffer {
			public:
				inline IndexBuffer ();
				inline virtual ~IndexBuffer ();
				inline void draw (VertexBuffer & vertexBuffer);
		};

		class MeshBuffer {
			public:
				inline MeshBuffer ();
				inline virtual ~MeshBuffer ();
				inline VertexBuffer & vertexBuffer () { return _vertexBuffer; }
				inline const VertexBuffer & vertexBuffer () const { return _vertexBuffer; }
				inline IndexBuffer & indexBuffer () { return _indexBuffer; }
				inline const IndexBuffer & indexBuffer () const { return _indexBuffer; }
				inline void draw ();
			private:
				VertexBuffer _vertexBuffer;
				IndexBuffer _indexBuffer;
		};

#define printOpenGLError(X) printOglError ((X), __FILE__, __LINE__)

		/// Returns 1 if an OpenGL error occurred, 0 otherwise.
		static int printOglError (const std::string & msg, const char * file, int line) {
			GLenum glErr;
			int    retCode = 0;
			glErr = glGetError ();
			while (glErr != GL_NO_ERROR) {
				printf ("glError in file %s @ line %d: %s - %s\n", file, line, gluErrorString(glErr), msg.c_str ());
				retCode = 1;
				glErr = glGetError ();
			}
			return retCode;
		}

		static void checkGLExceptions () {
			GLenum glerr = glGetError ();
			switch (glerr) {
				case GL_INVALID_ENUM:
					throw MorphoGraphics::GL::Exception ("Invalid Enum");
				case GL_INVALID_VALUE:
					throw MorphoGraphics::GL::Exception ("Invalid Value");
				case GL_INVALID_OPERATION:
					throw MorphoGraphics::GL::Exception ("Invalid Operation");
				case GL_STACK_OVERFLOW:
					throw MorphoGraphics::GL::Exception ("Stack Overflow");
				case GL_STACK_UNDERFLOW:
					throw MorphoGraphics::GL::Exception ("Stack Underflow");
				case GL_OUT_OF_MEMORY:
					throw MorphoGraphics::GL::Exception ("Out of Memory");
				case GL_TABLE_TOO_LARGE:
					throw MorphoGraphics::GL::Exception ("Table Too Large");
				case GL_NO_ERROR:
					break;
				default:
					throw MorphoGraphics::GL::Exception ("Other Error");
			}
		}

		inline Shader::Shader (const std::string & name, GLuint type) {
			_id = glCreateShader (type);
			_name = name;
			_type = type;
			_filename = "";
			_source = "";
		}

		inline Shader::~Shader () {
			if (_id != 0)
				glDeleteShader (_id);
		}

		inline void Shader::setSource (const std::string & source) {
			_source = source;
		}

		inline void Shader::compile () {
			const GLchar * tmp = _source.c_str();
			glShaderSource (_id, 1, &tmp, NULL);
			glCompileShader (_id);
			printOpenGLError ("Compiling Shader " + name ());  // Check for OpenGL errors
			GLint shaderCompiled;
			glGetShaderiv (_id, GL_COMPILE_STATUS, &shaderCompiled);
			printOpenGLError ("Compiling Shader " + name ());  // Check for OpenGL errors
			if (!shaderCompiled)
				throw MorphoGraphics::GL::Exception ("Error: shader not compiled. Info. Log.:\n" + infoLog () + "\nSource:\n" + _source);
		}

		inline void Shader::loadFromFile (const std::string & filename) {
			_filename = filename;
			std::ifstream in (_filename.c_str ());
			if (!in)
				throw MorphoGraphics::GL::Exception ("Error loading shader source file: " + _filename);
			std::string source;
			char c[2];
			c[1]='\0';
			while (in.get (c[0]))
				source.append (c);
			in.close ();
			setSource (source);
		}

		inline void Shader::reload () {
			if (_filename != "") {
				loadFromFile (std::string (_filename));
				compile ();
			}
		}

		inline std::string Shader::infoLog () {
			std::string infoLogStr = "";
			int infologLength = 0;
			glGetShaderiv (_id, GL_INFO_LOG_LENGTH, &infologLength);
			printOpenGLError ("Gathering Shader InfoLog Length for " + name ());
			if (infologLength > 0) {
				GLchar *str = new GLchar[infologLength];
				int charsWritten  = 0;
				glGetShaderInfoLog (_id, infologLength, &charsWritten, str);
				printOpenGLError ("Gathering Shader InfoLog for " + name ());
				infoLogStr  = std::string (str);
				delete [] str;
			}
			return infoLogStr;
		}

		inline Program::Program (const std::string & name) :
			_id (glCreateProgram ()),
			_name (name) {}

		inline Program::~Program () {
			glDeleteProgram (_id);
		}

		inline void Program::attach (Shader * shader) {
			glAttachShader (_id, shader->id());
			_shaders.push_back (shader);
		}

		inline void Program::detach (Shader * shader) {
			for (unsigned int i = 0; i < _shaders.size (); i++)
				if (_shaders[i]->id () == shader->id ())
					glDetachShader (_id, shader->id());
		}

		inline void Program::link () {
			glLinkProgram (_id);
			printOpenGLError ("Linking Program " + name ());
			GLint linked;
			glGetProgramiv (_id, GL_LINK_STATUS, &linked);
			if (!linked)
				throw MorphoGraphics::GL::Exception ("Shaders not linked: " + infoLog ());
		}

		inline void Program::use () {
			glUseProgram (_id);
		}

		inline void Program::stop () {
			glUseProgram (0);
		}

		inline std::string Program::infoLog () {
			std::string infoLogStr = "";
			int infologLength = 0;
			glGetProgramiv (_id, GL_INFO_LOG_LENGTH, &infologLength);
			printOpenGLError ("Gathering Shader InfoLog for Program " + name ());
			if (infologLength > 0) {
				GLchar *str = new GLchar[infologLength];
				int charsWritten  = 0;
				glGetProgramInfoLog (_id, infologLength, &charsWritten, str);
				printOpenGLError ("Gathering Shader InfoLog for Program " + name ());
				infoLogStr  = std::string (str);
				delete [] str;
			}
			return infoLogStr;
		}

		inline GLint Program::getUniformLocation (const std::string & uniformName) {
			const GLchar * cname = uniformName.c_str ();
			GLint loc = glGetUniformLocation (_id, cname);
			if (loc == -1)
				throw MorphoGraphics::GL::Exception (std::string ("Program Error: No such uniform named ") + uniformName);
			printOpenGLError ("Wrong Uniform Variable [" + uniformName + "] for Program [" + name () + "]");
			return loc;
		}

		inline void Program::setUniform1f (GLint location, float value) {
			use ();
			glUniform1f (location, value);
		}

		inline void Program::setUniform1f (const std::string & name, float value) {
			use ();
			glUniform1f (getUniformLocation (name), value);
		}

		inline void Program::setUniform2f (GLint location, float value0, float value1) {
			use ();
			glUniform2f (location, value0, value1);
		}

		inline void Program::setUniform2f (const std::string & name, float value0, float value1) {
			use ();
			glUniform2f (getUniformLocation (name), value0, value1);
		}

		inline void Program::setUniform3f (GLint location, float value0, float value1, float value2) {
			use ();
			glUniform3f (location, value0, value1, value2);
		}

		inline void Program::setUniform3f (const std::string & name, float value0, float value1, float value2) {
			use ();
			glUniform3f (getUniformLocation (name), value0, value1, value2);
		}

		inline void Program::setUniform4f (GLint location, float value0, float value1, float value2, float value3) {
			use ();
			glUniform4f (location, value0, value1, value2, value3);
		}

		inline void Program::setUniform4f (const std::string & name, float value0, float value1, float value2, float value3) {
			use ();
			glUniform4f (getUniformLocation (name), value0, value1, value2, value3);
		}

		inline void Program::setUniformMatrix4fv (GLint location, const float * values) {
			use ();
			glUniformMatrix4fv (location, 1, GL_FALSE, values);
		}

		inline void Program::setUniformMatrix4fv (const std::string & name, const float * values) {
			use ();
			setUniformMatrix4fv (getUniformLocation (name), values);
		}

		inline void Program::setUniformNf (GLint location, unsigned int numValues, const float * values) {
			use ();
			switch (numValues) {
				case 1: glUniform1f (location, values[0]); break;
				case 2: glUniform2f (location, values[0], values[1]); break;
				case 3: glUniform3f (location, values[0], values[1], values[2]); break;
				case 4: glUniform4f (location, values[0], values[1], values[2], values[3]); break;
				default: throw MorphoGraphics::GL::Exception ("Program Error: Wrong number of values to set for uniform float array.");
			}
		}

		inline void Program::setUniformNf (const std::string & name, unsigned int numValues, const float * values) {
			use ();
			GLint loc = getUniformLocation (name);
			switch (numValues) {
				case 1: glUniform1f (loc, values[0]); break;
				case 2: glUniform2f (loc, values[0], values[1]); break;
				case 3: glUniform3f (loc, values[0], values[1], values[2]); break;
				case 4: glUniform4f (loc, values[0], values[1], values[2], values[3]); break;
				default: throw MorphoGraphics::GL::Exception ("Wrong number of values to set for uniform float array " + name + ".");
			}
		}

		inline void Program::setUniform1i (GLint location, int value) {
			use ();
			glUniform1i (location, value);
		}

		inline void Program::setUniform1i (const std::string & name, int value) {
			use ();
			glUniform1i (getUniformLocation (name), value);
		}

		inline void Program::setUniformNi (GLint location, unsigned int numValues, const int * values) {
			use ();
			switch (numValues) {
				case 1: glUniform1i (location, values[0]); break;
				case 2: glUniform2i (location, values[0], values[1]); break;
				case 3: glUniform3i (location, values[0], values[1], values[2]); break;
				case 4: glUniform4i (location, values[0], values[1], values[2], values[3]); break;
				default: throw MorphoGraphics::GL::Exception ("Program Error: Wrong number of values to set for uniform int array.");
			}
		}

		inline void Program::setUniformNi (const std::string & name, unsigned int numValues, const int * values) {
			use ();
			GLint loc = getUniformLocation (name);
			switch (numValues) {
				case 1: glUniform1i (loc, values[0]); break;
				case 2: glUniform2i (loc, values[0], values[1]); break;
				case 3: glUniform3i (loc, values[0], values[1], values[2]); break;
				case 4: glUniform4i (loc, values[0], values[1], values[2], values[3]); break;
				default: throw MorphoGraphics::GL::Exception ("Program Error: Wrong number of values to set for uniform int array " + name + ".");
			}
		}

		inline void Program::reload () {
			for (unsigned int i = 0; i < _shaders.size (); i++) {
				_shaders[i]->reload ();
				attach (_shaders[i]);
			}
			link ();
		}

		inline Program * Program::genVFProgram (const std::string & name,
																						const std::string & vertexShaderFilename,
																						const std::string & fragmentShaderFilename) {
			Program * p = new Program (name);
			Shader * vs = new Shader (name + " Vertex Shader", GL_VERTEX_SHADER);
			Shader * fs = new Shader (name + " Fragment Shader",GL_FRAGMENT_SHADER);
			vs->loadFromFile (vertexShaderFilename);
			vs->compile ();
			p->attach(vs);
			fs->loadFromFile (fragmentShaderFilename);
			fs->compile ();
			p->attach(fs);
			p->link();
			//p->use ();
			return p;
		}

		inline Program * Program::genVGFProgram (const std::string & name,
																						 const std::string & vertexShaderFilename,
																						 const std::string & geometryShaderFilename,
																						 const std::string & fragmentShaderFilename) {
			Program * p = new Program (name);
			Shader * vs = new Shader (name + " Vertex Shader", GL_VERTEX_SHADER);
			Shader * gs = new Shader (name + " Geometry Shader", GL_GEOMETRY_SHADER);
			Shader * fs = new Shader (name + " Fragment Shader",GL_FRAGMENT_SHADER);
			vs->loadFromFile (vertexShaderFilename);
			vs->compile ();
			p->attach(vs);
			gs->loadFromFile (geometryShaderFilename);
			gs->compile ();
			p->attach(gs);
			fs->loadFromFile (fragmentShaderFilename);
			fs->compile ();
			p->attach(fs);
			p->link();
			//p->use ();
			return p;
		}


		inline Framebuffer::Framebuffer () {
			glGenFramebuffers (1, &_id);
		}

		inline Framebuffer::~Framebuffer () {
			glDeleteFramebuffers (1, &_id);
		}

		inline void Framebuffer::bind (GLenum target) {
			glBindFramebuffer (target, _id);
		}

		inline Buffer::Buffer () {
			_target = GL_ARRAY_BUFFER;
			_size = 0;
			_usage = GL_STATIC_DRAW;
			_mode = GL_TRIANGLES;
			glGenBuffers (1, &_id);
		}

		inline Buffer::~Buffer () {}

		void Buffer::bind () {
			glBindBuffer (_target, _id);
		}

		void Buffer::unbind () {
			glBindBuffer (_target, 0);
		}

		void Buffer::setData (GLsizei size, const GLvoid * data) {
			_size = size;
			bind ();
			try {
				glBufferData (_target, _size, data, _usage);
				checkGLExceptions ();
			} catch (MorphoGraphics::GL::Exception e) {
				throw MorphoGraphics::GL::Exception ("Error initializing buffer data: "
																						 + e.msg ());
			}
		}

		void Buffer::clear () {
			glDeleteBuffers (1, &_id);
			glGenBuffers (1, &_id);
		}

		//-------------------------------------------------
		// Vertex Buffer
		//-------------------------------------------------

		VertexBuffer::VertexBuffer (): Buffer () {
			setTarget (GL_ARRAY_BUFFER);
			setUsage (GL_STATIC_DRAW);
			setMode (GL_POINTS);
		}

		VertexBuffer::~VertexBuffer () {}

		void VertexBuffer::preDraw () {
			bind ();
			try {
				checkGLExceptions ();
			} catch (MorphoGraphics::GL::Exception e) {
				throw MorphoGraphics::GL::Exception ("Error before enabling VertexAttribArray: "
																						 + e.msg ());
			}
			try {
				glEnableVertexAttribArray (0);
				glEnableVertexAttribArray (1);
				glEnableVertexAttribArray (2);
				checkGLExceptions ();
			} catch (MorphoGraphics::GL::Exception e) {
				throw MorphoGraphics::GL::Exception ("Error enabling VertexAttribArray: "
																						 + e.msg ());
			}
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), 0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)12);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9*sizeof (float), (void*)24);
		}

		void VertexBuffer::draw () {
			preDraw ();
			glDrawArrays (mode (), 0, size ()/9*sizeof(float));
			postDraw ();
		}

		void VertexBuffer::postDraw () {
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);
			unbind ();
		}

		//-------------------------------------------------
		// Index Buffer
		//-------------------------------------------------

		IndexBuffer::IndexBuffer (): Buffer () {
			setTarget (GL_ELEMENT_ARRAY_BUFFER);
			setUsage (GL_STATIC_DRAW);
			setMode (GL_TRIANGLES);
		}

		IndexBuffer::~IndexBuffer () {}

		void IndexBuffer::draw (VertexBuffer & vertexBuffer) {
			vertexBuffer.preDraw ();
			bind ();
			glDrawElements (mode (), size ()/sizeof (unsigned int), GL_UNSIGNED_INT, 0);
			unbind ();
			vertexBuffer.postDraw ();
		}

		//-------------------------------------------------
		// Mesh Buffer
		//-------------------------------------------------

		MeshBuffer::MeshBuffer () {
		}

		MeshBuffer::~MeshBuffer () {
		}

		void MeshBuffer::draw () {
			_indexBuffer.draw (_vertexBuffer);
		}
	};
};

#endif // BLADESDK_GL_MODULE

// Some Emacs-Hints -- please don't remove:
//
//  Local Variables:
//  mode:C++
//  tab-width:4
//  End:
