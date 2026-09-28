#ifndef AMR_CRITERIA_HPP_
#define AMR_CRITERIA_HPP_

#include "forest.hpp"
#include <vector>

struct Hydro_ader;
struct MHD_ader;
struct SD_Solution;

//edge: this block's six face spacing ratios (lohner_edge_ratios), or nullptr for
//the interior-only score.
double lohner_score(SD_Solution W, int var, const double* edge = nullptr);
//false under SPD_LOHNER_INTERIOR=1: the interior-only score of commit 6339d79 and
//before, which is the A/B reference. main.cpp's block-size guard and Mesh's
//pre-tag exchange both read it.
bool lohner_edge_on();
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
