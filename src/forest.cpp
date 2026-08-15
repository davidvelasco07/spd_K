#include "spd_k.hpp"
#include "forest.hpp"

static constexpr double _TOL = 1e-12;

void dimension_set_origin(dimension &d, double origin){
    auto shift = [&](Matrix &M){
        auto h = Kokkos::create_mirror_view(M);
        Kokkos::deep_copy(h, M);
        for(int j=0; j<(int)h.extent(0); j++)
        for(int i=0; i<(int)h.extent(1); i++)
            h(j,i) += origin;
        Kokkos::deep_copy(M, h);
    };
    auto shiftv = [&](Vector &V){
        auto h = Kokkos::create_mirror_view(V);
        Kokkos::deep_copy(h, V);
        for(int i=0; i<(int)h.extent(0); i++) h(i) += origin;
        Kokkos::deep_copy(V, h);
    };
    shift(d.sd_faces);
    shift(d.sd_centers);
    shiftv(d.fv_faces);
    shiftv(d.fv_centers);
}

static dimension make_block_dim(int dim, int NB, int p, double lim_lo, double lim_hi,
                                double *x_fp, bool active){
    double width = lim_hi - lim_lo;
    dimension d(dim, NB, NB, active ? p : 0, 0, width, x_fp, active);
    dimension_set_origin(d, lim_lo);
    d.h = width / NB;
    return d;
}

dimension block_dimension_x(const MeshBlock &b, int NB, int p, double *x_fp, bool ax){
    return make_block_dim(_x_, NB, p, b.lim[_x_][0], b.lim[_x_][1], x_fp, ax);
}
dimension block_dimension_y(const MeshBlock &b, int NB, int p, double *x_fp, bool ay){
    return make_block_dim(_y_, NB, p, b.lim[_y_][0], b.lim[_y_][1], x_fp, ay);
}
dimension block_dimension_z(const MeshBlock &b, int NB, int p, double *x_fp, bool az){
    return make_block_dim(_z_, NB, p, b.lim[_z_][0], b.lim[_z_][1], x_fp, az);
}

BlockForest BlockForest::uniform_grid(
    int ndim_, bool ax, bool ay, bool az,
    int NBx, int NBy, int NBz,
    int nbx, int nby, int nbz,
    const double lim[3][2],
    const int bc_[3]){
    BlockForest forest;
    forest.ndim = ndim_;
    forest.active[0] = ax; forest.active[1] = ay; forest.active[2] = az;
    forest.NB[0] = NBx; forest.NB[1] = NBy; forest.NB[2] = NBz;
    forest.N_base[0] = nbx; forest.N_base[1] = nby; forest.N_base[2] = nbz;
    for(int d=0; d<3; d++){
        forest.bc[d] = bc_[d];
        forest.domain_lim[d][0] = lim[d][0];
        forest.domain_lim[d][1] = lim[d][1];
    }

    double block_len[3];
    for(int d=0; d<3; d++)
        block_len[d] = (lim[d][1]-lim[d][0]) / forest.N_base[d];

    double h0[3];
    for(int d=0; d<3; d++)
        h0[d] = block_len[d] / forest.NB[d];

    int ib = 0;
    int iz_range = az ? nbz : 1;
    int iy_range = ay ? nby : 1;
    int ix_range = nbx;
    for(int iz=0; iz<iz_range; iz++)
    for(int iy=0; iy<iy_range; iy++)
    for(int ix=0; ix<ix_range; ix++){
        MeshBlock b;
        b.ib = ib++;
        b.level = 0;
        b.logical[0] = ix; b.logical[1] = ay ? iy : 0; b.logical[2] = az ? iz : 0;
        for(int d=0; d<3; d++){
            int g = (d==0 ? ix : (d==1 ? iy : iz));
            if(!forest.active[d]) g = 0;
            b.lim[d][0] = lim[d][0] + g * block_len[d];
            b.lim[d][1] = b.lim[d][0] + block_len[d];
            b.h[d] = h0[d];
        }
        forest.blocks.push_back(b);
    }
    forest.rebuild_neighbors();
    return forest;
}

BlockForest::BlockKey BlockForest::block_key(int ib) const {
    const MeshBlock &b = blocks[ib];
    return {b.level, b.logical[0], b.logical[1], b.logical[2]};
}

int BlockForest::sub_face_index_logical(int coarse_ib, int fine_ib, int dim) const {
    const MeshBlock &coarse = blocks[coarse_ib];
    const MeshBlock &fine = blocks[fine_ib];
    int idx = 0, stride = 1;
    for(int d=0; d<3; d++){
        if(d == dim || !active[d]) continue;
        int bit = fine.logical[d] - 2 * coarse.logical[d];
        idx += bit * stride;
        stride *= 2;
    }
    return idx;
}

void BlockForest::rebuild_neighbors(){
    for(int ib=0; ib<(int)blocks.size(); ib++){
        blocks[ib].ib = ib;
        for(int d=0; d<3; d++)
        for(int s=0; s<2; s++)
            blocks[ib].neighbors[d][s].clear();
    }

    std::map<BlockKey,int> addr;
    for(int ib=0; ib<(int)blocks.size(); ib++)
        addr[block_key(ib)] = ib;

    std::vector<int> levels;
    for(const auto &b : blocks){
        if(std::find(levels.begin(), levels.end(), b.level) == levels.end())
            levels.push_back(b.level);
    }
    std::sort(levels.begin(), levels.end());

    for(int ib=0; ib<(int)blocks.size(); ib++){
        MeshBlock &block = blocks[ib];
        int L = block.level;
        for(int dim=0; dim<3; dim++){
            if(!active[dim]) continue;
            for(int side=0; side<2; side++){
                int step = side==0 ? -1 : 1;
                int N_L_d = N_at(L, dim);
                int new_k = block.logical[dim] + step;
                bool out_of_bounds = (new_k < 0 || new_k >= N_L_d);
                if(out_of_bounds && bc[dim] != _periodic_){
                    block.neighbors[dim][side].push_back({-1, NEIGH_BC, -1});
                    continue;
                }
                if(out_of_bounds) new_k = (new_k % N_L_d + N_L_d) % N_L_d;

                int same_logical[3] = {block.logical[0], block.logical[1], block.logical[2]};
                same_logical[dim] = new_k;

                BlockKey same_key = {L, same_logical[0], same_logical[1], same_logical[2]};
                auto it = addr.find(same_key);
                if(it != addr.end()){
                    block.neighbors[dim][side].push_back({it->second, NEIGH_SAME, -1});
                    continue;
                }

                bool found = false;
                for(int li=(int)levels.size()-1; li>=0; li--){
                    int L2 = levels[li];
                    if(L2 >= L) continue;
                    int shift = L - L2;
                    int coarser_logical[3];
                    for(int k=0; k<3; k++) coarser_logical[k] = same_logical[k] >> shift;
                    BlockKey ckey = {L2, coarser_logical[0], coarser_logical[1], coarser_logical[2]};
                    it = addr.find(ckey);
                    if(it != addr.end()){
                        int sub = (shift==1) ? sub_face_index_logical(it->second, ib, dim) : -1;
                        block.neighbors[dim][side].push_back({it->second, NEIGH_COARSER, sub});
                        found = true;
                        break;
                    }
                }
                if(found) continue;

                std::vector<NeighborEntry> finer_here;
                for(int L2 : levels){
                    if(L2 <= L) continue;
                    int shift = L2 - L;
                    int shift_n = 1 << shift;
                    int face_k;
                    if(side == 0) face_k = block.logical[dim] * shift_n - 1;
                    else face_k = (block.logical[dim] + 1) * shift_n;
                    int N_L2_d = N_at(L2, dim);
                    if(face_k < 0 || face_k >= N_L2_d){
                        if(bc[dim] == _periodic_)
                            face_k = (face_k % N_L2_d + N_L2_d) % N_L2_d;
                        else continue;
                    }
                    int n_sub_lev = 1;
                    for(int d=0; d<3; d++) if(d!=dim && active[d]) n_sub_lev *= shift_n;
                    for(int sub_idx=0; sub_idx<n_sub_lev; sub_idx++){
                        int fine_logical[3];
                        int sub_k_counter = 0;
                        for(int kk=0; kk<3; kk++){
                            if(kk == dim){
                                fine_logical[kk] = face_k;
                                continue;
                            }
                            if(!active[kk]){ fine_logical[kk] = 0; continue; }
                            int offset = (sub_idx / (1 << sub_k_counter)) % shift_n;
                            fine_logical[kk] = block.logical[kk] * shift_n + offset;
                            sub_k_counter++;
                        }
                        BlockKey fkey = {L2, fine_logical[0], fine_logical[1], fine_logical[2]};
                        it = addr.find(fkey);
                        if(it != addr.end()){
                            int report_sub = (shift==1) ? sub_idx : -1;
                            finer_here.push_back({it->second, NEIGH_FINER, report_sub});
                        }
                    }
                }
                if(!finer_here.empty()){
                    block.neighbors[dim][side].insert(
                        block.neighbors[dim][side].end(),
                        finer_here.begin(), finer_here.end());
                    continue;
                }
                block.neighbors[dim][side].push_back({-1, NEIGH_BC, -1});
            }
        }
    }
    build_fast_paths();
}

void BlockForest::build_fast_paths(){
    dropped_faces = 0;
    for(int d=0; d<3; d++)
    for(int s=0; s<2; s++){
        same_jb[d][s].clear();
        face_groups[d][s] = FaceGroups{};
    }

    int Nb = (int)blocks.size();
    int ntrans = 0;
    for(int d=0; d<3; d++) if(active[d]) ntrans++;
    int n_sub = ntrans > 0 ? (1 << (ntrans - 1)) : 1;

    for(int dim=0; dim<3; dim++){
        if(!active[dim]) continue;
        for(int side=0; side<2; side++){
            FaceGroups &g = face_groups[dim][side];
            bool all_same = true;
            std::vector<int> jbs(Nb);
            for(int ib=0; ib<Nb; ib++){
                const auto &entries = blocks[ib].neighbors[dim][side];
                if(entries.size()==1 && entries[0].rel==NEIGH_SAME){
                    jbs[ib] = entries[0].jb;
                    g.same_ib.push_back(ib);
                    g.same_jb.push_back(entries[0].jb);
                } else all_same = false;

                if(entries.size()==1 && entries[0].rel==NEIGH_BC)
                    g.bc_ib.push_back(ib);
                else if(entries.size()==1 && entries[0].rel==NEIGH_COARSER){
                    g.co_ib.push_back(ib);
                    g.co_jb.push_back(entries[0].jb);
                    g.co_sub.push_back(entries[0].sub >= 0 ? entries[0].sub : 0);
                } else if(!entries.empty()){
                    bool all_finer = true;
                    for(const auto &e : entries) if(e.rel != NEIGH_FINER) all_finer = false;
                    //A face group needs all 2^(ndim-1) children to be usable.
                    //Partial groups do occur: adapt refines, then derefines,
                    //then calls enforce_2to1_balance, and each of the first two
                    //rebuilds neighbours, so this runs on intermediate forests
                    //that are still unbalanced (a face can front one child at
                    //L+1 and two at L+2). Those groups are dropped here and the
                    //final, balanced rebuild replaces them before any exchange
                    //or flux correction reads them.
                    bool grouped = false;
                    //This branch also takes a plain SAME face (it is not BC and
                    //not COARSER), so only a face that really fronts finer
                    //neighbours can be a drop.
                    bool any_finer = false;
                    for(const auto &e : entries)
                        if(e.rel == NEIGH_FINER) any_finer = true;
                    if(all_finer && (int)entries.size()==n_sub){
                        std::vector<int> row(n_sub, -1);
                        for(const auto &e : entries)
                            row[e.sub >= 0 ? e.sub : 0] = e.jb;
                        bool ok = true;
                        for(int v : row) if(v<0) ok = false;
                        if(ok){
                            g.fi_ib.push_back(ib);
                            g.fi_jb.push_back(row);
                            grouped = true;
                        }
                    }
                    //Counted, not silently discarded: see dropped_faces.
                    if(any_finer && !grouped) dropped_faces++;
                }
            }
            if(all_same) same_jb[dim][side] = jbs;
            for(size_t k=0; k<g.co_sub.size(); k++){
                int s = g.co_sub[k];
                g.coarser_by_sub[s].first.push_back(g.co_ib[k]);
                g.coarser_by_sub[s].second.push_back(g.co_jb[k]);
            }

        }
    }
}

std::vector<MeshBlock> BlockForest::refine_block_mutate(int ib){
    MeshBlock parent = blocks[ib];
    int new_level = parent.level + 1;
    double child_h[3];
    for(int d=0; d<3; d++) child_h[d] = parent.h[d] * 0.5;

    std::vector<MeshBlock> children;
    int iz_range = active[2] ? 2 : 1;
    int iy_range = active[1] ? 2 : 1;
    int ix_range = 2;
    for(int iz=0; iz<iz_range; iz++)
    for(int iy=0; iy<iy_range; iy++)
    for(int ix=0; ix<ix_range; ix++){
        MeshBlock c;
        c.level = new_level;
        c.logical[0] = 2*parent.logical[0] + ix;
        c.logical[1] = active[1] ? 2*parent.logical[1] + iy : 0;
        c.logical[2] = active[2] ? 2*parent.logical[2] + iz : 0;
        for(int d=0; d<3; d++){
            double lo = parent.lim[d][0], hi = parent.lim[d][1];
            double mid = 0.5*(lo+hi);
            int sel = (d==0 ? ix : (d==1 ? iy : iz));
            if(!active[d]){ c.lim[d][0]=lo; c.lim[d][1]=hi; }
            else c.lim[d][0] = sel==0 ? lo : mid, c.lim[d][1] = sel==0 ? mid : hi;
            c.h[d] = child_h[d];
        }
        children.push_back(c);
    }
    blocks.erase(blocks.begin()+ib);
    blocks.insert(blocks.end(), children.begin(), children.end());
    return children;
}

void BlockForest::refine_blocks(const std::vector<int> &ibs){
    std::vector<BlockKey> keys;
    keys.reserve(ibs.size());
    for(int i : ibs) keys.push_back(block_key(i));
    for(const BlockKey &key : keys){
        std::map<BlockKey,int> id_map;
        for(int ib=0; ib<(int)blocks.size(); ib++) id_map[block_key(ib)] = ib;
        auto it = id_map.find(key);
        if(it != id_map.end()) refine_block_mutate(it->second);
    }
    rebuild_neighbors();
}

MeshBlock BlockForest::derefine_block_mutate(const std::vector<int> &ibs){
    int n_sib = 1;
    for(int d=0; d<3; d++) if(active[d]) n_sib *= 2;
    if((int)ibs.size() != n_sib)
        throw std::runtime_error("derefine expects 2^ndim siblings");

    int level = blocks[ibs[0]].level;
    int parent_logical[3];
    for(int k=0; k<3; k++) parent_logical[k] = blocks[ibs[0]].logical[k] >> 1;

    //The parent is built as the bounding box of these blocks, so a group that
    //is not actually a sibling set would silently produce an oversized block
    //sitting at the wrong place in the tree. Refuse instead.
    for(int i : ibs){
        bool ok = blocks[i].level == level;
        for(int k=0; k<3; k++)
            if((blocks[i].logical[k] >> 1) != parent_logical[k]) ok = false;
        if(!ok)
            throw std::runtime_error("derefine group is not a sibling set");
    }

    MeshBlock parent;
    parent.level = level - 1;
    parent.logical[0] = parent_logical[0];
    parent.logical[1] = parent_logical[1];
    parent.logical[2] = parent_logical[2];
    for(int d=0; d<3; d++){
        parent.lim[d][0] = blocks[ibs[0]].lim[d][0];
        parent.lim[d][1] = blocks[ibs[0]].lim[d][1];
        for(int i : ibs){
            parent.lim[d][0] = std::min(parent.lim[d][0], blocks[i].lim[d][0]);
            parent.lim[d][1] = std::max(parent.lim[d][1], blocks[i].lim[d][1]);
        }
        parent.h[d] = 2.0 * blocks[ibs[0]].h[d];
    }

    std::vector<int> sorted = ibs;
    std::sort(sorted.begin(), sorted.end(), std::greater<int>());
    for(int i : sorted) blocks.erase(blocks.begin()+i);
    blocks.push_back(parent);
    return parent;
}

std::vector<std::vector<BlockForest::BlockKey>>
BlockForest::keys_of(const std::vector<std::vector<int>> &groups) const {
    std::vector<std::vector<BlockKey>> key_groups;
    for(const auto &group : groups){
        std::vector<BlockKey> kg;
        for(int i : group) kg.push_back(block_key(i));
        key_groups.push_back(kg);
    }
    return key_groups;
}

void BlockForest::derefine_blocks(const std::vector<std::vector<int>> &groups){
    derefine_blocks_keys(keys_of(groups));
}

//A derefinement that enforce_2to1_balance() would immediately undo is not a
//derefinement: it is a block teardown, a neighbour rebuild and a refine cascade
//that end where they started, repeated every adapt forever. Refuse it up front.
//
//athenak does the same thing at the tree (MeshBlockTree::Derefine walks the
//neighbour directions and returns without acting if the child facing this block
//is itself refined). spd does NOT -- it derefines, then rebalances, and so has
//the same latent churn; it just never parks in a state that triggers it.
//
//The test is written against the same face-neighbour table balance uses, so it
//predicts exactly what balance would do: the group collapses to level L-1, so a
//neighbour outside the group deeper than L is a 2-level jump.
//
//It judges the forest as it stands, which makes it conservative for a region
//coarsening by more than one level at once: the deepest groups go first, and a
//shallower group blocked only by a neighbour that is itself about to derefine
//has to wait for the next adapt. Coarsening then peels one level per adapt
//interval, which is the rate athenak imposes anyway via refinement_interval.
bool BlockForest::derefine_allowed(const std::vector<int> &ibs) const {
    if(ibs.empty()) return false;
    const int L = blocks[ibs[0]].level;
    for(int ib : ibs){
        for(int dim=0; dim<3; dim++){
            if(!active[dim]) continue;
            for(int side=0; side<2; side++)
                for(const auto &e : blocks[ib].neighbors[dim][side]){
                    if(e.jb < 0) continue;
                    if(std::find(ibs.begin(), ibs.end(), e.jb) != ibs.end()) continue;
                    if(blocks[e.jb].level > L) return false;
                }
        }
    }
    return true;
}

void BlockForest::derefine_blocks_keys(
        const std::vector<std::vector<BlockKey>> &key_groups){
    derefine_refused = 0;
    int n_sib = 1;
    for(int d=0; d<3; d++) if(active[d]) n_sib *= 2;
    for(const auto &kg : key_groups){
        std::map<BlockKey,int> id_map;
        for(int ib=0; ib<(int)blocks.size(); ib++) id_map[block_key(ib)] = ib;
        std::vector<int> ibs;
        for(const BlockKey &k : kg){
            auto it = id_map.find(k);
            if(it != id_map.end()) ibs.push_back(it->second);
        }
        //Groups are tagged against the mesh as it stood before this adapt, and
        //the refine pass runs first: a sibling can have been refined away in the
        //meantime. A partly present group is no longer a derefinable set, so drop
        //it and let the next regrid retag.
        if((int)ibs.size() != n_sib) continue;
        //Last line of defence: a caller that skipped the filter still cannot
        //push the forest into a state balance has to repair.
        if(!derefine_allowed(ibs)){ derefine_refused++; continue; }
        derefine_block_mutate(ibs);
    }
    rebuild_neighbors();
}

int BlockForest::enforce_2to1_balance(){
    int total = 0;
    while(true){
        std::vector<int> to_refine;
        for(int ib=0; ib<(int)blocks.size(); ib++){
            for(int dim=0; dim<3; dim++){
                if(!active[dim]) continue;
                for(int side=0; side<2; side++){
                    for(const auto &e : blocks[ib].neighbors[dim][side]){
                        if(e.jb < 0) continue;
                        if(blocks[e.jb].level - blocks[ib].level > 1)
                            to_refine.push_back(ib);
                    }
                }
            }
        }
        if(to_refine.empty()) return total;
        std::sort(to_refine.begin(), to_refine.end());
        to_refine.erase(std::unique(to_refine.begin(), to_refine.end()), to_refine.end());
        refine_blocks(to_refine);
        total += (int)to_refine.size();
    }
}

int BlockForest::target_level(const MeshBlock &block,
                              const std::vector<RefinementRegion> &regions) const {
    int target = 0;
    for(const auto &spec : regions){
        bool overlap = true;
        if(block.lim[0][1] <= spec.xmin + _TOL || block.lim[0][0] >= spec.xmax - _TOL) overlap = false;
        if(active[1] && (block.lim[1][1] <= spec.ymin + _TOL || block.lim[1][0] >= spec.ymax - _TOL)) overlap = false;
        if(active[2] && (block.lim[2][1] <= spec.zmin + _TOL || block.lim[2][0] >= spec.zmax - _TOL)) overlap = false;
        if(overlap) target = std::max(target, spec.level);
    }
    return target;
}

void BlockForest::refine_to_levels(const std::vector<RefinementRegion> &regions){
    while(true){
        std::vector<int> to_refine;
        for(int ib=0; ib<(int)blocks.size(); ib++)
            if(target_level(blocks[ib], regions) > blocks[ib].level)
                to_refine.push_back(ib);
        if(to_refine.empty()) return;
        refine_blocks(to_refine);
    }
}
