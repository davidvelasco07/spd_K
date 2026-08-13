#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string.h>
#include <list>

#if __has_include("H5Cpp.h")
#include "H5Cpp.h"
using namespace H5;
#endif

#include "define.hpp"
#include "structs.hpp"
#include "fv_structs.hpp"
#include "global.hpp"
#include "muscl.hpp"
#include "prototypes.hpp"
#include "tasklist/task_list.hpp"
#include "driver.hpp"
#include "induction_ader.hpp"
#include "forest.hpp"
#include "hydro_ader.hpp"
#include "mhd.hpp"
#include "amr_criteria.hpp"
#include "mesh.hpp"
#include "hydro_mesh.hpp"

using namespace std;