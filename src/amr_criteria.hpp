#ifndef AMR_CRITERIA_HPP_
#define AMR_CRITERIA_HPP_

#include "forest.hpp"
#include <vector>

struct Hydro_ader;
struct MHD_ader;
struct SD_Solution;

double lohner_score(SD_Solution W, int var);
double pressure_gradient_score(SD_Solution W);
double shear_score(SD_Solution W);
double bfield_mean(SD_Solution W);
double bfield_lohner_score(SD_Solution W, double bref);
double trouble_fraction(Hydro_ader& blk);
double trouble_fraction(MHD_ader& blk);

//The pack views are how the scores get computed without a per-block host copy:
//one primitives launch and one score launch over the whole pack. Pass the same
//pv.U_sp / pv.W_sp the advance uses.
void tag_blocks(BlockForest& forest, std::vector<Hydro_ader>& blocks,
                SD_Solution U_pack, SD_Solution W_pack,
                std::vector<int>& to_refine,
                std::vector<std::vector<int>>& to_derefine,
                int max_level, int criterion);
void tag_blocks(BlockForest& forest, std::vector<MHD_ader>& blocks,
                SD_Solution U_pack, SD_Solution W_pack,
                std::vector<int>& to_refine,
                std::vector<std::vector<int>>& to_derefine,
                int max_level, int criterion);

#endif
