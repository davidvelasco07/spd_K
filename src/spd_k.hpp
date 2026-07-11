#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>
#include <fstream>
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
#include "induction_ader.hpp"
#include "hydro_ader.hpp"

using namespace std;