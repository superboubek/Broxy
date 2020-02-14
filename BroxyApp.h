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

#ifndef BROXYAPP_H
#define BROXYAPP_H

#define GLEW_STATIC 1
#include <GL/glew.h>

#include <QApplication>
#include <QCoreApplication>
#include <QMainWindow>
#include <QDockWidget>
#include <QComboBox>
#include <QGroupBox>

#include "BroxyViewer.h"

class BroxyApp : public QApplication {
	Q_OBJECT
	
public:
	BroxyApp (int & argc, char ** argv);
	virtual ~BroxyApp ();
	int run ();
	
public slots:
	void openMesh ();
	void saveProxy ();
	void clean ();
	
private slots:
	void remesh ();
	void generateFrameField ();
	
private:
	QMainWindow * m_mainWindow;
	QDockWidget * m_dock;
	QWidget * m_paramsWidget;
	QGroupBox * m_proxyMeshingGroupBox;
	QComboBox * m_remeshingComboBox;
	QComboBox * m_symmetryComboBox;
	BroxyViewer * m_viewer;
};

#endif // BROXYAPP_H