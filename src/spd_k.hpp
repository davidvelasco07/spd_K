#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <list>

#include "H5Cpp.h"

#include "define.hpp"
#include "structs.hpp"
#include "fv_structs.hpp"
#include "global.hpp"
#include "prototypes.hpp"
#include "induction_ader.hpp"
#include "hydro_ader.hpp"

using namespace std;
using namespace H5;