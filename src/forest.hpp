#ifndef FOREST_HPP_
#define FOREST_HPP_

#include "define.hpp"
#include "structs.hpp"
#include <map>
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>

//Neighbor relation tags (spd/amr/tree.py).
enum NeighborRelation {
    NEIGH_SAME = 0,
    NEIGH_FINER,
    NEIGH_COARSER,
    NEIGH_BC
};

struct NeighborEntry {
    int jb = -1;              // neighbor block index, -1 for BC
    NeighborRelation rel = NEIGH_BC;
    int sub = -1;             // sub-face index for FINER/COARSER; -1 if N/A
};

struct MeshBlock {
    int ib = 0;
    int level = 0;
    int logical[3] = {0, 0, 0};   // (ix, iy, iz) at this level
    double lim[3][2] = {};        // physical bounds [lo, hi] per dim
    double h[3] = {};             // per-element physical size per dim
    // neighbors[dim][side=0|1] -> list of entries (multi-entry on finer faces)
    std::array<std::array<std::vector<NeighborEntry>, 2>, 3> neighbors;
};

//Rectangular refinement region (AthenaK-style <refinementN> blocks).
struct RefinementRegion {
    int level = 0;
    double xmin = -1e300, xmax = 1e300;
    double ymin = -1e300, ymax = 1e300;
    double zmin = -1e300, zmax = 1e300;
};

//Precomputed face groups for batched boundary exchange (spd face_groups).
struct FaceGroups {
    std::vector<int> same_ib, same_jb;
    std::vector<int> bc_ib;
    std::vector<int> co_ib, co_jb, co_sub;
    std::map<int, std::pair<std::vector<int>, std::vector<int>>> coarser_by_sub;
    std::vector<int> fi_ib;
    std::vector<std::vector<int>> fi_jb; // [n_fi][n_sub]
};

class BlockForest {
  public:
    int ndim = 3;
    bool active[3] = {true, true, true};
    int NB[3] = {1, 1, 1};           // elements per block per dim
    int N_base[3] = {1, 1, 1};       // blocks per dim at coarsest level
    int bc[3] = {_periodic_, _periodic_, _periodic_};
    double domain_lim[3][2] = {};    // full domain bounds per dim
    std::vector<MeshBlock> blocks;

    //Fast paths rebuilt after every neighbor table update
    std::array<std::array<std::vector<int>, 2>, 3> same_jb; // nullptr -> empty = mixed
    std::array<std::array<FaceGroups, 2>, 3> face_groups;

    //Faces build_fast_paths could not put in any group: a face fronting finer
    //neighbours that do not form a complete set of 2^(ndim-1) children, or one
    //fronting a mix of relations. Such a face gets no ghost fill and no flux
    //correction -- silently. That is legal only on the intermediate, still
    //unbalanced forests inside adapt(); on the forest a step actually runs on
    //it means the level jump at that face is unhandled.
    int dropped_faces = 0;

    int Nblocks() const { return (int)blocks.size(); }
    int max_level() const {
        int m = 0;
        for(const auto &b : blocks) m = std::max(m, b.level);
        return m;
    }

    static BlockForest uniform_grid(
        int ndim_,
        bool ax, bool ay, bool az,
        int NBx, int NBy, int NBz,
        int nbx, int nby, int nbz,
        const double lim[3][2],
        const int bc_[3]);

    void rebuild_neighbors();
    void refine_blocks(const std::vector<int> &ibs);
    void derefine_blocks(const std::vector<std::vector<int>> &groups);
    //True when collapsing this sibling group to its parent leaves the forest
    //2:1 balanced. False means enforce_2to1_balance() would refine it straight
    //back, so the derefinement is pure churn -- see derefine_allowed's comment.
    bool derefine_allowed(const std::vector<int> &ibs) const;
    //Groups derefine_blocks_keys skipped for that reason, last call.
    int derefine_refused = 0;
    int enforce_2to1_balance();
    void refine_to_levels(const std::vector<RefinementRegion> &regions);

    //Logical key for solution remap across adapt
    using BlockKey = std::tuple<int, int, int, int>; // level, lx, ly, lz
    BlockKey block_key(int ib) const;

    //Refining invalidates block indices, so groups that outlive a refine must
    //be carried as keys.
    std::vector<std::vector<BlockKey>>
        keys_of(const std::vector<std::vector<int>> &groups) const;
    void derefine_blocks_keys(const std::vector<std::vector<BlockKey>> &key_groups);

    int sub_face_index_logical(int coarse_ib, int fine_ib, int dim) const;

  private:
    int bid(int bx, int by, int bz, int nbx, int nby) const {
        return bx + nbx * (by + nby * bz);
    }
    int N_at(int L, int dim) const { return N_base[dim] * (1 << L); }
    void build_fast_paths();
    int target_level(const MeshBlock &block,
                     const std::vector<RefinementRegion> &regions) const;
    std::vector<MeshBlock> refine_block_mutate(int ib);
    MeshBlock derefine_block_mutate(const std::vector<int> &ibs);
};

//Shift a block-local dimension to physical coordinates [lim_lo, lim_hi].
void dimension_set_origin(dimension &d, double origin);

//Build per-block dimension objects from a forest block.
dimension block_dimension_x(const MeshBlock &b, int NB, int p, double *x_fp, bool ax);
dimension block_dimension_y(const MeshBlock &b, int NB, int p, double *x_fp, bool ay);
dimension block_dimension_z(const MeshBlock &b, int NB, int p, double *x_fp, bool az);

struct FV_Solution;

void restrict_face_fv_sub(FV_Solution C, FV_Solution F, int dim,
                          int cface, int fface, int cx, int cy, int cz,
                          int Ncx, int Ncy, int Ncz, int nx, int ny, int nz);

template<typename Block>
void forest_exchange_fp(BlockForest&, std::vector<Block>&, int dim);
template<typename Block>
void forest_exchange_sd(BlockForest&, std::vector<Block>&,
                        SD_Solution Block::*, int dim, bool cf_prolong=true);
template<typename Block>
void forest_sync_face_B(BlockForest&, std::vector<Block>&,
                        SD_Solution Block::*, int dim);
template<typename Block>
void correct_coarse_fine_flux(BlockForest&, std::vector<Block>&, int dim);
template<typename Block>
void correct_coarse_fine_emf(BlockForest&, std::vector<Block>&, int dim);
template<typename Block>
void correct_coarse_fine_fv_flux(BlockForest&, std::vector<Block>&, int dim);

template<typename Block>
void correct_coarse_fine_fv_emf(BlockForest&, std::vector<Block>&, int dim);
template<typename Block>
void symmetrize_same_level_fv_flux(BlockForest&, std::vector<Block>&, int dim);
template<typename Block>
void forest_exchange_fv(BlockForest&, std::vector<Block>&,
                        FV_Solution Block::*, int dim);
template<typename Block>
void forest_exchange_fv_same(BlockForest&, std::vector<Block>&,
                             FV_Solution Block::*, int dim);
template<typename Block>
void forest_exchange_fv_max(BlockForest&, std::vector<Block>&,
                            FV_Solution Block::*, int dim);

#endif
