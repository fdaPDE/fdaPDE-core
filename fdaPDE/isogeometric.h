#ifndef __FDAPDE_ISOGEOMETRIC_MODULE_H__
#define __FDAPDE_ISOGEOMETRIC_MODULE_H__

namespace fdapde {
    struct iso_tag {};
}

#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "fields.h"
#include "nurbs.h"
#include "geometry.h"

//isogeometric analysis
#include "src/isogeometric/iso_algorithms.h"
#include "src/geometry/iso_cell.h"
#include "src/geometry/iso_segment.h"
#include "src/geometry/iso_square.h"
#include "src/geometry/iso_cube.h"
#include "src/geometry/iso_mesh.h"
#include "src/isogeometric/dof_constraints.h"
#include "src/isogeometric/dof_handler.h"
#include "src/isogeometric/iso_integration.h"
#include "src/assembly.h"
#include "src/isogeometric/iso_assembler_base.h"
#include "src/isogeometric/iso_bilinear_form_assembler.h"
#include "src/isogeometric/iso_linear_form_assembler.h"
#include "src/isogeometric/iso_space.h"
// weak forms
#include "src/isogeometric/iso_objects.h"



#endif  // ___FDAPDE_ISOGEOMETRIC_MODULE_H__

