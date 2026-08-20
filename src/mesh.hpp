#ifndef MESH_HPP_
#define MESH_HPP_

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

#include "forest.hpp"
#include "amr_criteria.hpp"

//Scoped Kokkos profiling region. A no-op unless a tool is loaded
//(KOKKOS_TOOLS_LIBS), and it is what attributes kernel launches to a phase
//when measuring per-block launch overhead.
struct Region {
    explicit Region(const char* n){ Kokkos::Profiling::pushRegion(n); }
    ~Region(){ Kokkos::Profiling::popRegion(); }
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;
};

//Block-forest mesh driver: uniform multiblock and mixed-level AMR (hydro +
//true-2D MHD). Derives from PhysicsModule and registers mesh-orchestrated
//tasks into Driver.
//Phase timing for the advance, enabled by SPD_PHASE_TIMES=1.
//
//Why this exists rather than a profiler: nsys on apollo (2022.1.3) cannot export
//its own report ("Version number is invalid"), and no Kokkos kp_*.so is built
//there. This needs no tooling, works identically on CPU and GPU, and answers the
//one question that matters for the per-block dispatch cost -- which phase is
//paying it. Each scope fences before and after, so it SERIALISES the pipeline:
//it is a diagnostic, never something to leave enabled.
struct PhaseTimes {
    static bool on(){
        static const bool b = getenv("SPD_PHASE_TIMES")!=nullptr;
        return b;
    }
    std::map<std::string,double> t;
    std::map<std::string,long>   n;
    void add(const std::string& k, double s){ t[k]+=s; n[k]++; }
    void report(const char* tag) const {
        if(!on() || !Master || t.empty()) return;
        double tot = 0.0;
        for(const auto& kv : t) tot += kv.second;
        std::vector<std::pair<double,std::string>> v;
        for(const auto& kv : t) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        std::cout<<"\nphase times ("<<tag<<"), total "<<std::fixed
                 <<std::setprecision(3)<<tot<<" s over fenced scopes:"<<std::endl;
        for(const auto& e : v)
            std::cout<<"  "<<std::setw(8)<<std::setprecision(3)<<e.first<<" s "
                     <<std::setw(5)<<std::setprecision(1)<<(100*e.first/tot)<<"%  "
                     <<std::setw(9)<<n.at(e.second)<<" calls  "<<e.second<<std::endl;
        std::cout<<std::defaultfloat;
    }
};

struct PhaseScope {
    PhaseTimes* p;
    const char* k;
    bool on;
    std::chrono::steady_clock::time_point t0;
    PhaseScope(PhaseTimes* p_, const char* k_) : p(p_), k(k_), on(PhaseTimes::on()){
        if(on){ Kokkos::fence(); t0 = std::chrono::steady_clock::now(); }
    }
    ~PhaseScope(){
        if(!on) return;
        Kokkos::fence();
        p->add(k, std::chrono::duration<double>(
                      std::chrono::steady_clock::now()-t0).count());
    }
};
#define PHASE(name) PhaseScope _phase_scope_(&phase_times_, name)

template<typename Block>
struct Mesh : public PhysicsModule {
    PhaseTimes phase_times_;

    BlockForest forest;
    int NBx, NBy, NBz;
    int nblocks;
    int n_ader;

    CommHelper comm_;
    int p_;
    double *x_, *w_, *x_sp_, *x_fp_;
    double nu_ = 0.0, beta_ = 0.0;

    std::vector<Block> blocks;
    std::vector<dimension> Xd, Yd, Zd;
    dimension Xg, Yg, Zg;
    SD_Solution W_glob;

    //All blocks' evolution arrays live here, one allocation per array name
    //with a leading block axis, so a phase can be one kernel over the mesh
    //instead of one kernel per block.
    BlockPack pack;

    //The one per-run spectral-difference operator set, aliased into every block
    //instead of rebuilt per block per adapt (see SDOperators).
    SDOperators ops_;

    //Block geometry is a pure function of the block's forest key: level plus
    //logical index feed x_dim_for_block & co., and everything else they read is
    //run-constant. So a block a regrid did not touch can keep the dimension
    //objects it already has, and a regrid pays for geometry only on the blocks
    //that actually changed. Held as keys parallel to Xd/Yd/Zd; a dimension is
    //just Views, so a hit is a refcount copy instead of 4 device allocations,
    //4 host mirrors and 4 deep_copies per direction.
    std::vector<BlockForest::BlockKey> geom_keys_;

    static constexpr bool is_hydro = std::is_same_v<Block, Hydro_ader>;
    static constexpr bool is_mhd   = std::is_same_v<Block, MHD_ader>;

    Mesh(
        BlockForest f,
        CommHelper comm,
        int p,
        dimension X_dim,
        dimension Y_dim,
        dimension Z_dim,
        int _NBx,
        int _NBy,
        int _NBz,
        double* x,
        double* w,
        double* x_sp,
        double* x_fp,
        double nu = 0.0,
        double beta = 0.0
    ) : forest(std::move(f)), Xg(X_dim), Yg(Y_dim), Zg(Z_dim),
        comm_(comm), p_(p), x_(x), w_(w), x_sp_(x_sp), x_fp_(x_fp),
        nu_(nu), beta_(beta)
    {
        NBx=_NBx; NBy=_NBy; NBz=_NBz;
        nblocks = forest.Nblocks();
        if(is_mhd && forest.max_level()>0 && cfg.active[_z_]){
            if(Master) std::cout<<"ERROR: mixed-level AMR for 3D MHD is not implemented yet"
                                  <<" (Stage 4 supports true-2D static refinement)"<<std::endl;
            exit(1);
        }
        if(forest.max_level()>0 && cfg.integrator==_integrator_ader_){
            if(Master) std::cout<<"ERROR: ADER is not supported with mixed-level AMR (use rk2/rk3)"<<std::endl;
            exit(1);
        }
        build_block_solvers();
        initial_refine();

        n_ader = blocks[0].n_ader;

        this->Dt = blocks[0].Dt;
        for(int b=1;b<nblocks;b++)
            this->Dt = std::min(this->Dt, blocks[b].Dt);

        init_W_glob(X_dim, Y_dim, Z_dim, x_fp);
        if(Master)
            std::cout<<"forest blocks = "<<nblocks<<" max_level = "<<forest.max_level()
                <<" NB = ("<<NBx<<","<<NBy<<","<<NBz<<") dt = "<<this->Dt<<std::endl;
        Write_outputs();
    }

    Block make_block(const dimension& Xdim, const dimension& Ydim, const dimension& Zdim,
                     int ib, bool run_ic){
        if constexpr (is_hydro)
            return Hydro_ader(comm_,p_,Xdim,Ydim,Zdim,x_,w_,x_sp_,x_fp_,nu_,beta_,false,
                              &pack, ib, &ops_, run_ic);
        else
            return MHD_ader(comm_,p_,Xdim,Ydim,Zdim,x_,w_,x_sp_,x_fp_,false,
                            &pack, ib, &ops_, run_ic);
    }

    //Uniform unigrid blocks use the legacy global-offset dimension layout so
    //per-cell arithmetic matches the pre-forest multiblock driver exactly.
    dimension x_dim_for_block(const MeshBlock& b, int p, double* x_fp) const {
        if(b.level==0)
            return dimension(_x_, Xg.N_global, NBx, cfg.active[_x_] ? p : 0,
                             b.logical[0]*NBx, Xg.L, x_fp, cfg.active[_x_]);
        return block_dimension_x(b, NBx, cfg.active[_x_] ? p : 0, x_fp, cfg.active[_x_]);
    }
    dimension y_dim_for_block(const MeshBlock& b, int p, double* x_fp) const {
        if(b.level==0)
            return dimension(_y_, Yg.N_global, NBy, cfg.active[_y_] ? p : 0,
                             b.logical[1]*NBy, Yg.L, x_fp, cfg.active[_y_]);
        return block_dimension_y(b, NBy, cfg.active[_y_] ? p : 0, x_fp, cfg.active[_y_]);
    }
    dimension z_dim_for_block(const MeshBlock& b, int p, double* x_fp) const {
        if(b.level==0)
            return dimension(_z_, Zg.N_global, NBz, cfg.active[_z_] ? p : 0,
                             b.logical[2]*NBz, Zg.L, x_fp, cfg.active[_z_]);
        return block_dimension_z(b, NBz, cfg.active[_z_] ? p : 0, x_fp, cfg.active[_z_]);
    }

    //run_ic=false means "a regrid is rebuilding these blocks": the initial
    //conditions are skipped because transfer_from_snapshot is about to overwrite
    //every block's state anyway.
    void build_block_solvers(bool run_ic=true){
        if(!ops_.built) build_sd_operators(ops_, p_, x_sp_, x_fp_);
        //Salvage the previous geometry, keyed by block, before Xd/Yd/Zd go.
        std::map<BlockForest::BlockKey, std::array<dimension,3>> old_geom;
        for(size_t i=0; i<geom_keys_.size() && i<Xd.size(); i++)
            old_geom.emplace(geom_keys_[i],
                             std::array<dimension,3>{Xd[i], Yd[i], Zd[i]});
        blocks.clear(); Xd.clear(); Yd.clear(); Zd.clear();
        geom_keys_.clear();
        nblocks = forest.Nblocks();
        //Drop the previous pack before the new one is sized: adapt() changes
        //the block count, so every array is reallocated with a new leading
        //extent and the old slices must not keep it alive.
        pack.reset(nblocks);
        for(int ib=0; ib<nblocks; ib++){
            const MeshBlock& b = forest.blocks[ib];
            const BlockForest::BlockKey key = forest.block_key(ib);
            auto it = old_geom.find(key);
            if(it != old_geom.end()){
                Xd.push_back(it->second[0]);
                Yd.push_back(it->second[1]);
                Zd.push_back(it->second[2]);
            }else{
                Xd.emplace_back(x_dim_for_block(b, p_, x_fp_));
                Yd.emplace_back(y_dim_for_block(b, p_, x_fp_));
                Zd.emplace_back(z_dim_for_block(b, p_, x_fp_));
            }
            geom_keys_.push_back(key);
            blocks.push_back(make_block(Xd[ib], Yd[ib], Zd[ib], ib, run_ic));
        }
        build_geometry_pack();
        build_pack_views();
        build_rk_pairs();
        build_neighbor_tables();
        build_xchg_tables();
        build_emf_corner_table();
    }

    //Per-block geometry that batched kernels need by block index: element
    //size and the FV sub-grid coordinates. Blocks differ only in these, which
    //is exactly what lets one kernel span refinement levels.
    Vector hx_p, hy_p, hz_p;      //element size per block
    Matrix fvx_p, fvy_p, fvz_p;      //FV face coordinates per block
    Matrix fvxc_p, fvyc_p, fvzc_p;   //FV cell centres per block (SED stencil)

    //Whole-pack views of the arrays the batched phases touch, cached so the
    //hot loop does no map lookups. Rebuilt with the pack on every adapt.
    struct PackViews {
        SD_Solution U_ader_fp_x, U_ader_fp_y, U_ader_fp_z;
        SD_Solution F_ader_fp_x, F_ader_fp_y, F_ader_fp_z;
        SD_Solution U_sp, W_sp, W_cv, U_cv, U_ader_sp, U0_sp, T_sweep;
        SD_Solution T_fp_x, T_fp_y, T_fp_z;
        FV_Solution U_old, U_new, W_old, W_new, theta;
        FV_Solution flagged, cascade, troubles, theta_tmp;
        FV_Solution F1_x, F1_y, F1_z, F2_x, F2_y, F2_z;
        FV_Solution alpha_x, alpha_y, alpha_z;
        FV_Solution F_x, F_y, F_z;
        //MHD. The pack already HOLDS these (MHD_ader routes every array through
        //alloc() -> init_packed, exactly as Hydro_ader does); only the whole-pack
        //handles were missing, which is why the MHD advance was still a per-block
        //loop while every hydro phase had been batched.
        SD_Solution Bx_fp_x, By_fp_y, Bz_fp_z;
        SD_Solution B0x_fp_x, B0y_fp_y, B0z_fp_z;   //RK save of the face field
        SD_Solution Bxf, Byf, Bzf, TB_x, TB_y, TB_z;
        SD_Solution Ex_ep_yz, Ey_ep_zx, Ez_ep_xy;
        FV_Solution B_old_cv, B_new_cv;
        FV_Solution F0_x, F0_y, F0_z, E0x, E0y, E0z;
        FV_Solution U_old_fv, U_new_fv;
        FV_Solution Bx_old, Bx_new, By_old, By_new, Bz_old, Bz_new;
        FV_Solution F1m_x, F1m_y, F1m_z, F2m_x, F2m_y, F2m_z;
        FV_Solution E1x, E1y, E1z, E2x, E2y, E2z;
        FV_Solution W_fv, det_old, det_new, mhd_cascade;
        FV_Solution UCT1_x, UCT1_y, UCT1_z, UCT2_x, UCT2_y, UCT2_z;
        SD_Solution mhd_U_ader_sp;
    } pv;

    void build_pack_views(){
        pv.U_ader_fp_x = sd_pack_view(pack,"U_ader_fp_x");
        pv.U_ader_fp_y = sd_pack_view(pack,"U_ader_fp_y");
        pv.U_ader_fp_z = sd_pack_view(pack,"U_ader_fp_z");
        pv.F_ader_fp_x = sd_pack_view(pack,"F_ader_fp_x");
        pv.F_ader_fp_y = sd_pack_view(pack,"F_ader_fp_y");
        pv.F_ader_fp_z = sd_pack_view(pack,"F_ader_fp_z");
        pv.U_sp        = sd_pack_view(pack,"U_sp");
        pv.W_sp        = sd_pack_view(pack,"W_sp");
        pv.W_cv        = sd_pack_view(pack,"W_cv");
        pv.U_ader_sp   = sd_pack_view(pack,"U_ader_sp");
        pv.U0_sp       = sd_pack_view(pack,"U0_sp", true);   //RK only
        pv.T_sweep     = sd_pack_view(pack,"T_sweep");
        pv.U_cv        = sd_pack_view(pack,"U_cv");
        if constexpr (is_mhd){
            //T_fp_* is a hydro-only scratch; MHD's own path uses U_ader_fp_*.
            //Everything below the cfg.fallback guard is allocated only when the
            //MOOD cascade is on (mhd.hpp: `if(cfg.fallback){ ... }`), so asking
            //for it unconditionally aborts a job/fallback=false run -- which is
            //exactly what the hardened sd_pack_view reported.
            pv.Bx_fp_x  = sd_pack_view(pack,"Bx_fp_x");
            pv.By_fp_y  = sd_pack_view(pack,"By_fp_y");
            pv.Bz_fp_z  = sd_pack_view(pack,"Bz_fp_z");
            //The RK save of the staggered field. Allocated unconditionally by
            //MHD_ader (alloc -> init_packed), like everything above.
            pv.B0x_fp_x = sd_pack_view(pack,"B0x_fp_x");
            pv.B0y_fp_y = sd_pack_view(pack,"B0y_fp_y");
            pv.B0z_fp_z = sd_pack_view(pack,"B0z_fp_z");
            pv.Ex_ep_yz = sd_pack_view(pack,"Ex_ep_yz");
            pv.Ey_ep_zx = sd_pack_view(pack,"Ey_ep_zx");
            pv.Ez_ep_xy = sd_pack_view(pack,"Ez_ep_xy");
            pv.mhd_U_ader_sp = sd_pack_view(pack,"U_ader_sp");
            //Everything past here is allocated only when the MOOD cascade is on
            //(mhd.hpp: `if(cfg.fallback){ ... }`).
            if(!cfg.fallback) return;
            pv.Bxf      = sd_pack_view(pack,"Bxf");
            pv.Byf      = sd_pack_view(pack,"Byf");
            pv.Bzf      = sd_pack_view(pack,"Bzf");
            pv.TB_x     = sd_pack_view(pack,"TB_x");
            pv.TB_y     = sd_pack_view(pack,"TB_y");
            pv.TB_z     = sd_pack_view(pack,"TB_z");
            pv.B_old_cv = fv_pack_view(pack,"B_old_cv");
            pv.B_new_cv = fv_pack_view(pack,"B_new_cv");
            pv.F0_x     = fv_pack_view(pack,"F0_x");
            pv.F0_y     = fv_pack_view(pack,"F0_y");
            pv.F0_z     = fv_pack_view(pack,"F0_z");
            pv.E0x      = fv_pack_view(pack,"E0x");
            pv.E0y      = fv_pack_view(pack,"E0y");
            pv.E0z      = fv_pack_view(pack,"E0z");
            pv.U_old_fv = fv_pack_view(pack,"U_old_fv");
            pv.U_new_fv = fv_pack_view(pack,"U_new_fv");
            pv.Bx_old   = fv_pack_view(pack,"Bx_old");
            pv.Bx_new   = fv_pack_view(pack,"Bx_new");
            pv.By_old   = fv_pack_view(pack,"By_old");
            pv.By_new   = fv_pack_view(pack,"By_new");
            pv.Bz_old   = fv_pack_view(pack,"Bz_old");
            pv.Bz_new   = fv_pack_view(pack,"Bz_new");
            //MHD's level-1/2 flux arrays share names with hydro's F1_*/F2_*,
            //so they get their own handles to keep the two unambiguous.
            pv.F1m_x    = fv_pack_view(pack,"F1_x");
            pv.F1m_y    = fv_pack_view(pack,"F1_y");
            pv.F1m_z    = fv_pack_view(pack,"F1_z");
            pv.F2m_x    = fv_pack_view(pack,"F2_x");
            pv.F2m_y    = fv_pack_view(pack,"F2_y");
            pv.F2m_z    = fv_pack_view(pack,"F2_z");
            pv.E1x      = fv_pack_view(pack,"E1x");
            pv.E1y      = fv_pack_view(pack,"E1y");
            pv.E1z      = fv_pack_view(pack,"E1z");
            pv.E2x      = fv_pack_view(pack,"E2x");
            pv.E2y      = fv_pack_view(pack,"E2y");
            pv.E2z      = fv_pack_view(pack,"E2z");
            pv.W_fv     = fv_pack_view(pack,"W_fv");
            pv.det_old  = fv_pack_view(pack,"det_old");
            pv.det_new  = fv_pack_view(pack,"det_new");
            //`troubles` is assigned again further down, but that line is past the
            //`return` that ends this MHD branch -- so for MHD it was never
            //assigned at all and stayed a default-constructed (empty) view.
            //fv_pack_view's hardening cannot catch this: it aborts on a name
            //MISSING FROM THE PACK, not on a pv field nobody asked for. Writing
            //through the empty view is a straight segfault, which is how the
            //batched detection announced it.
            pv.troubles = fv_pack_view(pack,"troubles");
            pv.mhd_cascade = fv_pack_view(pack,"cascade");
            pv.UCT1_x   = fv_pack_view(pack,"UCT1_x");
            pv.UCT1_y   = fv_pack_view(pack,"UCT1_y");
            pv.UCT1_z   = fv_pack_view(pack,"UCT1_z");
            pv.UCT2_x   = fv_pack_view(pack,"UCT2_x");
            pv.UCT2_y   = fv_pack_view(pack,"UCT2_y");
            pv.UCT2_z   = fv_pack_view(pack,"UCT2_z");
            return;
        }
        //Hydro's FV/fallback arrays are allocated only under cfg.fallback
        //(hydro_ader.hpp), same as MHD's MOOD set above.
        if(!cfg.fallback) return;
        pv.T_fp_x      = sd_pack_view(pack,"T_fp_x");
        pv.T_fp_y      = sd_pack_view(pack,"T_fp_y");
        pv.T_fp_z      = sd_pack_view(pack,"T_fp_z");
        pv.F_x         = fv_pack_view(pack,"F_x");
        pv.F_y         = fv_pack_view(pack,"F_y");
        pv.F_z         = fv_pack_view(pack,"F_z");
        pv.U_old       = fv_pack_view(pack,"U_old");
        pv.U_new       = fv_pack_view(pack,"U_new");
        pv.W_old       = fv_pack_view(pack,"W_old");
        pv.W_new       = fv_pack_view(pack,"W_new");
        pv.theta       = fv_pack_view(pack,"theta");
        pv.flagged     = fv_pack_view(pack,"flagged");
        pv.troubles    = fv_pack_view(pack,"troubles");
        pv.alpha_x     = fv_pack_view(pack,"alpha_x");
        pv.alpha_y     = fv_pack_view(pack,"alpha_y");
        pv.alpha_z     = fv_pack_view(pack,"alpha_z");
        pv.theta_tmp   = fv_pack_view(pack,"theta_tmp");
        //cascade / F1 / F2 are nested one level deeper for hydro, under
        //cfg.mood_cascade -- the blend style does not allocate them.
        if(!cfg.mood_cascade) return;
        pv.cascade     = fv_pack_view(pack,"cascade");
        pv.F1_x        = fv_pack_view(pack,"F1_x");
        pv.F1_y        = fv_pack_view(pack,"F1_y");
        pv.F1_z        = fv_pack_view(pack,"F1_z");
        pv.F2_x        = fv_pack_view(pack,"F2_x");
        pv.F2_y        = fv_pack_view(pack,"F2_y");
        pv.F2_z        = fv_pack_view(pack,"F2_z");
    }

    void build_geometry_pack(){
        hx_p = Vector("pack_hx", nblocks);
        hy_p = Vector("pack_hy", nblocks);
        hz_p = Vector("pack_hz", nblocks);
        Vector_h hxh = setup_mirror(hx_p);
        Vector_h hyh = setup_mirror(hy_p);
        Vector_h hzh = setup_mirror(hz_p);
        for(int b=0;b<nblocks;b++){
            hxh(b)=Xd[b].h; hyh(b)=Yd[b].h; hzh(b)=Zd[b].h;
        }
        setup_push(hx_p,hxh); setup_push(hy_p,hyh); setup_push(hz_p,hzh);

        //One row per block of a per-direction FV coordinate array, so a batched
        //kernel can read its own block's geometry from a leading block index.
        auto pack_coord = [&](Matrix& M, const char* nm, std::vector<dimension>& D,
                              bool centers){
            int n = centers ? D[0].fv_ncells : D[0].fv_nfaces;
            M = Matrix(nm, nblocks, n);
            Matrix_h h = setup_mirror(M);
            for(int b=0;b<nblocks;b++){
                Vector& src = centers ? D[b].fv_centers : D[b].fv_faces;
                Vector_h f = setup_mirror(src);
                setup_pull(src, f);
                for(int i=0;i<n;i++) h(b,i) = f(i);
            }
            setup_push(M,h);
        };
        pack_coord(fvx_p,"pack_fvx",Xd,false);
        pack_coord(fvy_p,"pack_fvy",Yd,false);
        pack_coord(fvz_p,"pack_fvz",Zd,false);
        pack_coord(fvxc_p,"pack_fvxc",Xd,true);
        pack_coord(fvyc_p,"pack_fvyc",Yd,true);
        pack_coord(fvzc_p,"pack_fvzc",Zd,true);
    }

    void init_W_glob(dimension X_dim, dimension Y_dim, dimension Z_dim, double* x_fp){
        int M = forest.max_level();
        if(M>0){
            int sc = 1<<M;
            int fx = X_dim.N*sc, fy = Y_dim.N*sc, fz = Z_dim.N*sc;
            dimension Xf(_x_, X_dim.N_global*sc, fx, Xd[0].p, 0, X_dim.L, x_fp, cfg.active[_x_]);
            dimension Yf(_y_, Y_dim.N_global*sc, fy, Yd[0].p, 0, Y_dim.L, x_fp, cfg.active[_y_]);
            dimension Zf(_z_, Z_dim.N_global*sc, fz, Zd[0].p, 0, Z_dim.L, x_fp, cfg.active[_z_]);
            W_glob.init("W_cv",1,blocks[0].nvar,Zf,Yf,Xf,0,0,0);
            Write_dimensions(Xf, Yf, Zf);
        } else {
            W_glob.init("W_cv",1,blocks[0].nvar,Z_dim,Y_dim,X_dim,0,0,0);
        }
    }

    void recompute_dt(){
        this->Dt = 1e300;
        for(int b=0;b<nblocks;b++){
            if constexpr (is_hydro){
                compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
                blocks[b].cons_to_prim_cv();
                this->Dt = std::min(this->Dt,
                    compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h, nu_));
            } else {
                mhd_compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
                blocks[b].cons_to_prim_cv();
                this->Dt = std::min(this->Dt,
                    mhd_compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h));
            }
        }
    }

    SD_Solution& fp(int b, int dim){
        return dim==_x_ ? blocks[b].U_ader_fp_x
             : dim==_y_ ? blocks[b].U_ader_fp_y
                        : blocks[b].U_ader_fp_z;
    }

    void neighbors_uniform(int b, int dim, int &L, int &R, int &tL, int &tR){
        int nbx=forest.N_base[0], nby=forest.N_base[1];
        int nbz=forest.active[2] ? forest.N_base[2] : 1;
        int c[3]={b%nbx,(b/nbx)%nby,b/(nbx*nby)};
        int cl[3]={c[0],c[1],c[2]}, cr[3]={c[0],c[1],c[2]};
        cl[dim]=(c[dim]-1+(dim==0?nbx:dim==1?nby:nbz))%(dim==0?nbx:dim==1?nby:nbz);
        cr[dim]=(c[dim]+1)%(dim==0?nbx:dim==1?nby:nbz);
        L = cl[0]+nbx*(cl[1]+nby*cl[2]);
        R = cr[0]+nbx*(cr[1]+nby*cr[2]);
        tL = (c[dim]==0 && cfg.bc[dim]==_gradfree_) ? _gradfree_ : _periodic_;
        tR = (c[dim]==(dim==0?nbx:dim==1?nby:nbz)-1 && cfg.bc[dim]==_gradfree_) ? _gradfree_ : _periodic_;
    }

    //Per-block neighbour indices and side types on a uniform mesh, uploaded
    //once so the batched exchanges can read them inside the kernel instead of
    //the host re-deriving them per block per launch.
    IntVector nbrL_[3], nbrR_[3], typL_[3], typR_[3];

    //Views for Block::rk_state(), resolved once per pack rather than per stage:
    //the PackViews rationale applies -- the hot loop does no map lookups. Pairs
    //are (saved, live).
    std::vector<std::pair<SD_Solution,SD_Solution>> rk_pairs_;

    void build_rk_pairs(){
        rk_pairs_.clear();
        //Only when the RK state is really used -- the same condition
        //AssembleTasks registers TaskSaveState under -- because hydro allocates
        //U0_sp for RK only, and asking the pack for a name it does not have is
        //(correctly) fatal.
        if(!(cfg.integrator==_integrator_rk_ || is_mhd)) return;
        for(const auto& a : Block::rk_state())
            rk_pairs_.emplace_back(sd_pack_view(pack, a.first),
                                   sd_pack_view(pack, a.second));
    }

    void build_neighbor_tables(){
        for(int dim=0; dim<3; dim++){
            nbrL_[dim] = IntVector("nbrL", nblocks);
            nbrR_[dim] = IntVector("nbrR", nblocks);
            typL_[dim] = IntVector("typL", nblocks);
            typR_[dim] = IntVector("typR", nblocks);
            IntVector_h hL = setup_mirror(nbrL_[dim]);
            IntVector_h hR = setup_mirror(nbrR_[dim]);
            IntVector_h htL = setup_mirror(typL_[dim]);
            IntVector_h htR = setup_mirror(typR_[dim]);
            for(int b=0; b<nblocks; b++){
                int L,R,tL,tR;
                neighbors_uniform(b,dim,L,R,tL,tR);
                hL(b)=L; hR(b)=R; htL(b)=tL; htR(b)=tR;
            }
            setup_push(nbrL_[dim],hL); setup_push(nbrR_[dim],hR);
            setup_push(typL_[dim],htL); setup_push(typR_[dim],htR);
        }
    }

    SD_Solution& block_fp_dbg(int b, int dim){
        return dim==_x_ ? blocks[b].U_ader_fp_x
             : dim==_y_ ? blocks[b].U_ader_fp_y
                        : blocks[b].U_ader_fp_z;
    }

    //Ghost-fill transactions for the receiver-driven gather: one entry per
    //(receiving block, face) with the block it reads from. Rebuilt with the
    //forest on every adapt; the block index becomes a kernel axis rather than
    //a host loop. Same-level only for now -- a relation/sub column joins these
    //when mixed levels move onto the same path.
    struct XchgTable { IntVector recv, send, sub; int n = 0; };
    //One set per relation: same-level copies, coarse->fine prolongation, and
    //fine->coarse restriction. FINER expands to one transaction per fine
    //neighbour, each covering a disjoint quadrant of the coarse face, so the
    //"every ghost written exactly once" property holds across all three.
    XchgTable xt_[3][2], xtco_[3][2], xtfi_[3][2];

    static void push_table(XchgTable& t, const std::vector<int>& r,
                           const std::vector<int>& s, const std::vector<int>& sb){
        t = XchgTable{};
        if(r.empty()) return;
        t.n = (int)r.size();
        t.recv = IntVector("xchg_recv", r.size());
        t.send = IntVector("xchg_send", r.size());
        t.sub  = IntVector("xchg_sub",  r.size());
        auto hr = setup_mirror(t.recv);
        auto hs = setup_mirror(t.send);
        auto hb = setup_mirror(t.sub);
        for(size_t k=0; k<r.size(); k++){ hr(k)=r[k]; hs(k)=s[k]; hb(k)=sb[k]; }
        setup_push(t.recv, hr); setup_push(t.send, hs); setup_push(t.sub, hb);
    }

    void build_xchg_tables(){
        for(int dim=0; dim<3; dim++)
        for(int side=0; side<2; side++){
            xt_[dim][side] = XchgTable{};
            xtco_[dim][side] = XchgTable{};
            xtfi_[dim][side] = XchgTable{};
            if(!cfg.active[dim]) continue;
            std::vector<int> r, s, b, cr, cs, cb, fr, fs, fb;
            const auto& sj = forest.same_jb[dim][side];
            if(!sj.empty()){
                for(int ib=0; ib<nblocks; ib++){ r.push_back(ib); s.push_back(sj[ib]); b.push_back(0); }
            } else {
                const FaceGroups& g = forest.face_groups[dim][side];
                for(size_t k=0; k<g.same_ib.size(); k++){
                    r.push_back(g.same_ib[k]); s.push_back(g.same_jb[k]); b.push_back(0);
                }
                for(size_t k=0; k<g.co_ib.size(); k++){
                    cr.push_back(g.co_ib[k]); cs.push_back(g.co_jb[k]); cb.push_back(g.co_sub[k]);
                }
                for(size_t k=0; k<g.fi_ib.size(); k++)
                for(size_t t=0; t<g.fi_jb[k].size(); t++){
                    fr.push_back(g.fi_ib[k]); fs.push_back(g.fi_jb[k][t]); fb.push_back((int)t);
                }
            }
            push_table(xt_[dim][side],   r,  s,  b);
            push_table(xtco_[dim][side], cr, cs, cb);
            push_table(xtfi_[dim][side], fr, fs, fb);
        }
    }

    //--------------------------------------------------------------------------
    //Patch-corner transactions for the cascade's FV-lattice edge EMF: 6 ints per
    //entry, (src block, src i, src j, dst block, dst i, dst j). Pure topology,
    //so it is built with the forest rather than per step. What it is FOR is in
    //spread_fv_emf_corners_b (amr_boundary.cpp): the face correction leaves the
    //corner point of a refined patch multi-valued in the one block that touches
    //that point without sharing a face with any fine block.
    struct EmfCornerTable { IntVector t; int n = 0; };
    EmfCornerTable ect_;
    //Corner groups that are not the ordinary two-level, four-block case: a
    //corner where three levels meet, or more than four blocks at one point. The
    //same rule handles them, but a mesh that produces them is worth knowing
    //about, so the count is reported with the cf_flux line rather than left
    //silent -- see the note at the test itself for what sits just past it.
    int emf_corner_odd_ = 0;

    static bool no_emf_corner(){
        static const bool v = getenv("SPD_NO_EMF_CORNER") != nullptr;
        return v;
    }

    void build_emf_corner_table(){
        ect_ = EmfCornerTable{};
        emf_corner_odd_ = 0;
        if constexpr (!is_mhd) return;
        else {
        //Pure topology, so it serves BOTH lattices: the cascade's E0z (under
        //cfg.fallback) and the SD path's Ez_ep_xy (which exists either way). Do
        //not gate it on cfg.fallback -- that left the SD corner unfixed. 2D only,
        //matching both corrections (MHD_MOOD_update refuses 3D mixed levels).
        if(forest.max_level()==0 || cfg.active[_z_]) return;
        GHOST_LOCALS;
        const int Lmax = forest.max_level();
        //Key: the corner's index on the FINEST level's block lattice. Two blocks
        //at different levels that touch the same corner produce the same key by
        //integer shift -- no coordinate comparison and no tolerance, which is
        //what makes this independent of the arithmetic it is correcting. The
        //fold is applied only where the domain really wraps: on a non-periodic
        //side the high edge is NOT the low edge, and folding it would pair the
        //two ends of the domain.
        struct Touch { int ib, i, j, level; };
        std::map<std::pair<long,long>, std::vector<Touch>> corners;
        const long Nfx = (long)forest.N_base[_x_] << Lmax;
        const long Nfy = (long)forest.N_base[_y_] << Lmax;
        const bool wx = (forest.bc[_x_]==_periodic_), wy = (forest.bc[_y_]==_periodic_);
        for(int ib=0; ib<nblocks; ib++){
            const MeshBlock& b = forest.blocks[ib];
            const int sh = Lmax - b.level;
            //The same extents the face pass derives, from the same array.
            SD_Solution S = blocks[ib].W_cv;
            const int Ncx = (S.Nx-2*NGHx)*S.nx, Ncy = (S.Ny-2*NGHy)*S.ny;
            for(int cy=0; cy<2; cy++)
            for(int cx=0; cx<2; cx++){
                long kx = (long)(b.logical[_x_]+cx) << sh;
                long ky = (long)(b.logical[_y_]+cy) << sh;
                if(wx) kx %= Nfx;
                if(wy) ky %= Nfy;
                corners[{kx,ky}].push_back({ib, cx ? sghx+Ncx : sghx,
                                                cy ? sghy+Ncy : sghy, b.level});
            }
        }
        std::vector<int> e;
        for(const auto& kv : corners){
            const std::vector<Touch>& v = kv.second;
            int Lhi = -1; size_t src = 0;
            for(size_t k=0; k<v.size(); k++)
                if(v[k].level > Lhi){ Lhi = v[k].level; src = k; }
            bool jump = false;
            for(const Touch& t : v) if(t.level != Lhi) jump = true;
            //All one level: the point is already single-valued (the gate's
            //same-level control over faces carrying real flux is exactly 0), so
            //there is nothing to spread and nothing to perturb.
            if(!jump) continue;
            //A corner spanning three levels at once is legal -- 2:1 balance
            //constrains faces, not corners -- and the same rule handles it, but
            //it is worth knowing when a mesh produces one, because the one case
            //this construction genuinely cannot see lives next door: a block
            //whose corner falls at the MIDPOINT of a coarser block's face is
            //not a corner of that block, so it never joins its group. The face
            //pass owns that point (it is an interior point of the coarse face
            //line) and takes it from the face neighbour, which is the right
            //value unless a third, finer level touches it diagonally.
            int Llo = Lhi;
            for(const Touch& t : v) Llo = std::min(Llo, t.level);
            if(Lhi-Llo > 1 || v.size() > 4) emf_corner_odd_++;
            for(size_t k=0; k<v.size(); k++){
                //Blocks at the source's own level already agree with it bitwise;
                //writing between them could only reorder round-off.
                if(v[k].level == Lhi) continue;
                e.push_back(v[src].ib); e.push_back(v[src].i); e.push_back(v[src].j);
                e.push_back(v[k].ib);   e.push_back(v[k].i);   e.push_back(v[k].j);
            }
        }
        if(e.empty()) return;
        ect_.n = (int)e.size()/6;
        ect_.t = IntVector("emf_corner", e.size());
        auto h = setup_mirror(ect_.t);
        for(size_t k=0; k<e.size(); k++) h(k)=e[k];
        setup_push(ect_.t, h);
        }
    }

    //Batched exchanges (transaction tables) are the DEFAULT. The per-block
    //forest path is O(leaf count) launches per relation per direction, which on
    //an AMR forest is the whole cost: profiled on the fig-21 MUSCL lane (512^2
    //finest, 3 levels, 540 steps) the two FV halo exchanges alone were 6.42M of
    //7.81M launches -- 82% -- at 11,900 launches per step. Routing them through
    //the tables took the run from 7,810,161 launches / 131.5 s to 605,961 /
    //28.5 s: 12.9x fewer launches, 4.6x faster.
    //
    //SPD_OLD_XCHG=1 restores the per-block path; the two must agree exactly, so
    //it stays as the A/B reference behind SPD_EXCHANGE_CHECK.
    static bool new_xchg(){
        static bool v = getenv("SPD_OLD_XCHG") == nullptr;
        return v;
    }

    SD_Solution& fp_pack(int dim){
        return dim==_x_ ? pv.U_ader_fp_x
             : dim==_y_ ? pv.U_ader_fp_y
                        : pv.U_ader_fp_z;
    }

    //SPD_NO_PACK=1 routes a uniform forest through the per-block forest
    //exchange instead of the batched packed kernel. The two must agree
    //exactly; setting it isolates whether a discrepancy lives in the pack.
    static bool no_pack(){
        static bool v = getenv("SPD_NO_PACK") != nullptr;
        return v;
    }

    static bool exchange_check(){
        static bool v = getenv("SPD_EXCHANGE_CHECK") != nullptr;
        return v;
    }

    //Compare the forest result now in P against `other`, and print where the
    //two disagree.
    //
    //NaN is separated from an ordinary magnitude difference, because |a-b| is
    //NaN whenever either side is NaN and a NaN compares false against both
    //`==0` and `>worst`: it is counted as a difference while leaving the
    //worst-value tracker at its initial -1 indices. That combination -- many
    //differing entries, no worst entry -- means data that was never written,
    //not arithmetic that came out wrong, so it is reported first, with the
    //side that carries it and the receiving block named.
    //SPD_XCHK_FROM=<step> delays reporting until that step, so a mixed-level
    //comparison is not crowded out by the uniform steps before the first
    //regrid (the report is capped at a few occurrences).
    static int xchk_from(){
        static int v = getenv("SPD_XCHK_FROM") ? atoi(getenv("SPD_XCHK_FROM")) : 0;
        return v;
    }

    void report_exchange_diff(SD_Solution P, SD_Vector other, int dim){
        static int reported = 0;
        if(this->n_step < xchk_from()) return;
        auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.Vector);
        auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), other);
        double worst = 0.0;
        long nbad = 0, face_only = 0, corner = 0;
        long nan_active = 0;
        int nan_shown = 0;
        long nnan = 0, nan_forest = 0, nan_packed = 0, nan_both = 0;
        long nan_face = 0, nan_corner = 0;
        //Per-variable split. The last slot is the FV trouble-flag aggregate,
        //which the SD path never writes, so it can carry uninitialized data
        //without that meaning anything about the exchange.
        constexpr int MAXV = 16;
        long nan_var[MAXV] = {0}, bad_var[MAXV] = {0};
        int f[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
        int g[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
        const int gx=NGHx, gy=NGHy, gz=NGHz;
        for(size_t t=0; t<a.extent(0); t++)
        for(size_t v=0; v<a.extent(1); v++)
        for(size_t k=0; k<a.extent(2); k++)
        for(size_t j=0; j<a.extent(3); j++)
        for(size_t i=0; i<a.extent(4); i++)
        for(size_t kk=0; kk<a.extent(5); kk++)
        for(size_t jj=0; jj<a.extent(6); jj++)
        for(size_t ii=0; ii<a.extent(7); ii++){
            const double av = a(t,v,k,j,i,kk,jj,ii);
            const double bv = b(t,v,k,j,i,kk,jj,ii);
            //On the face proper (transverse indices all active) or only in the
            //transverse ghost corners/edges?
            bool tg = false;
            if(dim!=_x_ && ((int)i<gx || (int)i>=P.Nx-gx)) tg = true;
            if(dim!=_y_ && ((int)j<gy || (int)j>=P.Ny-gy)) tg = true;
            if(dim!=_z_ && ((int)k<gz || (int)k>=P.Nz-gz)) tg = true;
            const bool na = std::isnan(av), nb = std::isnan(bv);
            if(na || nb){
                nnan++;
                if((int)v < MAXV) nan_var[v]++;
                if(na && nb)   nan_both++;
                else if(na)    nan_forest++;
                else           nan_packed++;
                if(tg) nan_corner++; else nan_face++;
                if(g[0] < 0){
                    g[0]=(int)t; g[1]=(int)v; g[2]=(int)k; g[3]=(int)j;
                    g[4]=(int)i; g[5]=(int)kk; g[6]=(int)jj; g[7]=(int)ii;
                }
                continue;
            }
            const double d = std::abs(av - bv);
            if(d == 0.0) continue;
            nbad++;
            if((int)v < MAXV) bad_var[v]++;
            if(tg) corner++; else face_only++;
            if(d > worst){
                worst = d;
                f[0]=(int)t; f[1]=(int)v; f[2]=(int)k; f[3]=(int)j;
                f[4]=(int)i; f[5]=(int)kk; f[6]=(int)jj; f[7]=(int)ii;
            }
        }
        if((nbad || nnan) && nnan && Master)
            std::cout<<"        NaN by location: active="<<nan_active
                     <<" face-ghost="<<nan_face<<" corner-ghost="<<nan_corner
                     <<std::endl;
        if((nbad || nnan) && reported < 4 && Master){
            reported++;
            std::cout<<std::endl<<"[xchk] dim="<<dim<<" step="<<this->n_step;
            if(nnan){
                //The leading axis is block*n_ader + t_id, so it names the
                //receiving block directly.
                const int nad = P.n_ader;
                std::cout<<"\n       NaN entries = "<<nnan
                    <<"  (forest-only="<<nan_forest<<" packed-only="<<nan_packed
                    <<" both="<<nan_both<<")"
                    <<"  on-face = "<<nan_face<<"  transverse-ghost = "<<nan_corner
                    <<"\n       first NaN at (t="<<g[0]<<",var="<<g[1]<<",k="<<g[2]
                    <<",j="<<g[3]<<",i="<<g[4]<<",kk="<<g[5]<<",jj="<<g[6]<<",ii="<<g[7]<<")"
                    <<"  recv block = "<<(nad>0 ? g[0]/nad : -1)
                    <<" t_id = "<<(nad>0 ? g[0]%nad : -1)
                    <<"\n       first NaN values: forest="<<std::setprecision(17)
                    <<a(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7])
                    <<"  packed="<<b(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7])
                    <<std::setprecision(6);
            }
            std::cout<<"\n       differing entries = "<<nbad
                <<"  max|forest-packed| = "<<worst;
            if(nbad)
                std::cout<<"\n       worst at (t="<<f[0]<<",var="<<f[1]<<",k="<<f[2]
                    <<",j="<<f[3]<<",i="<<f[4]<<",kk="<<f[5]<<",jj="<<f[6]<<",ii="<<f[7]<<")"
                    <<"  forest="<<std::setprecision(17)
                    <<a(f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7])
                    <<"  packed="<<b(f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7])
                    <<std::setprecision(6);
            std::cout<<"\n       on-face entries = "<<face_only
                <<"   transverse-ghost (corner/edge) entries = "<<corner;
            const int nv = std::min((int)a.extent(1), MAXV);
            std::cout<<"\n       per-var  NaN:";
            for(int v=0; v<nv; v++) std::cout<<" ["<<v<<"]="<<nan_var[v];
            std::cout<<"\n       per-var diff:";
            for(int v=0; v<nv; v++) std::cout<<" ["<<v<<"]="<<bad_var[v];
            std::cout<<"\n       extents N=("<<P.Nz<<","<<P.Ny<<","<<P.Nx<<")"
                <<" n=("<<P.nz<<","<<P.ny<<","<<P.nx<<") NGH=("<<NGHz<<","<<NGHy<<","<<NGHx<<")"
                <<std::endl;
        }
    }

    //Same comparison for an FV field, which is (nvar, Nz, Ny, Nx): no ADER or
    //sub-point axis, and the halo is a slab rather than a face layer, so
    //"transverse ghost" here means a ghost cell of a direction other than the
    //one being exchanged. NaN is separated from magnitude for the reason given
    //on report_exchange_diff.
    void report_fv_diff(FV_Solution P, FV_Vector other, int dim){
        static int reported = 0;
        if(this->n_step < xchk_from()) return;
        auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.Vector);
        auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), other);
        double worst = 0.0;
        long nbad = 0, slab_only = 0, corner = 0, nnan = 0;
        long nan_active = 0, nan_face = 0, nan_corner = 0;
        int nan_shown = 0;
        int f[4] = {-1,-1,-1,-1};
        constexpr int MAXV = 16;
        long nan_var[MAXV] = {0}, bad_var[MAXV] = {0};
        const int gx = nGH_rt[_x_], gy = nGH_rt[_y_], gz = nGH_rt[_z_];
        for(size_t v=0; v<a.extent(0); v++)
        for(size_t k=0; k<a.extent(1); k++)
        for(size_t j=0; j<a.extent(2); j++)
        for(size_t i=0; i<a.extent(3); i++){
            const double av = a(v,k,j,i), bv = b(v,k,j,i);
            bool tg = false;
            if(dim!=_x_ && ((int)i<gx || (int)i>=P.Nx-gx)) tg = true;
            if(dim!=_y_ && ((int)j<gy || (int)j>=P.Ny-gy)) tg = true;
            if(dim!=_z_ && ((int)k<gz || (int)k>=P.Nz-gz)) tg = true;
            //The leading axis is block*n_var + var, so the histogram has to be
            //taken modulo n_var -- indexing it by the raw slot would only ever
            //cover the first few blocks and read as "no variable differs".
            const int var = (P.n_var>0) ? (int)v % P.n_var : 0;
            if(std::isnan(av) || std::isnan(bv)){
                nnan++;
                if(var < MAXV) nan_var[var]++;
                //Where the NaN sits: an unfilled ghost is a different bug from
                //an active cell that computed one. gz is 0 in 2D, so the z test
                //never fires there.
                const bool xg = ((int)i<gx || (int)i>=P.Nx-gx);
                const bool yg = ((int)j<gy || (int)j>=P.Ny-gy);
                const bool zg = (gz>0) && ((int)k<gz || (int)k>=P.Nz-gz);
                if(!xg && !yg && !zg) nan_active++;
                else if((int)(xg+yg+zg) >= 2) nan_corner++;
                else nan_face++;
                if(nan_shown < 8){
                    nan_shown++;
                    std::cout<<"        [nan] block "<<(P.n_var>0 ? (int)v/P.n_var : 0)
                             <<" var "<<var<<" (k,j,i)=("<<k<<","<<j<<","<<i<<")"
                             <<(!xg && !yg && !zg ? " ACTIVE"
                                : ((int)(xg+yg+zg)>=2 ? " corner-ghost" : " face-ghost"))
                             <<"  fwd="<<av<<" pkd="<<bv<<std::endl;
                }
                continue;
            }
            const double d = std::abs(av - bv);
            if(d == 0.0) continue;
            nbad++;
            if(var < MAXV) bad_var[var]++;
            if(tg) corner++; else slab_only++;
            if(d > worst){
                worst = d;
                f[0]=(int)v; f[1]=(int)k; f[2]=(int)j; f[3]=(int)i;
            }
        }
        if((nbad || nnan) && nnan && Master)
            std::cout<<"        NaN by location: active="<<nan_active
                     <<" face-ghost="<<nan_face<<" corner-ghost="<<nan_corner
                     <<std::endl;
        if((nbad || nnan) && reported < 4 && Master){
            reported++;
            //The leading axis is block*n_var + var, so it names the block.
            const int nv0 = P.n_var;
            std::cout<<std::endl<<"[fvchk] dim="<<dim<<" step="<<this->n_step
                <<"  NaN = "<<nnan<<"  differing entries = "<<nbad
                <<"  max|forest-packed| = "<<worst;
            if(nbad)
                std::cout<<"\n        worst at (slot="<<f[0]<<",k="<<f[1]
                    <<",j="<<f[2]<<",i="<<f[3]<<")"
                    <<"  block = "<<(nv0>0 ? f[0]/nv0 : -1)
                    <<" var = "<<(nv0>0 ? f[0]%nv0 : -1)
                    <<"  forest="<<std::setprecision(17)<<a(f[0],f[1],f[2],f[3])
                    <<"  packed="<<b(f[0],f[1],f[2],f[3])<<std::setprecision(6);
            std::cout<<"\n        in-slab entries = "<<slab_only
                <<"   transverse-ghost entries = "<<corner;
            const int nv = std::min(nv0, MAXV);
            std::cout<<"\n        per-var  NaN:";
            for(int v=0; v<nv; v++) std::cout<<" ["<<v<<"]="<<nan_var[v];
            std::cout<<"\n        per-var diff:";
            for(int v=0; v<nv; v++) std::cout<<" ["<<v<<"]="<<bad_var[v];
            std::cout<<"\n        extents N=("<<P.Nz<<","<<P.Ny<<","<<P.Nx<<")"
                <<" nGH=("<<gz<<","<<gy<<","<<gx<<") n_var="<<nv0
                <<std::endl;
        }
    }

    //All relations for one direction, as batched gathers over the
    //transaction tables. Physical boundaries stay a per-block call: there are
    //few of them and they need no neighbour.
    void gather_all_fp(int dim){
        SD_Solution P = fp_pack(dim);
        for(int side=0; side<2; side++){
            gather_fp_same(P, xt_[dim][side].recv, xt_[dim][side].send,
                           xt_[dim][side].n, dim, side);
            gather_fp_coarser(P, xtco_[dim][side].recv, xtco_[dim][side].send,
                              xtco_[dim][side].sub, xtco_[dim][side].n, dim, side, amr_P);
            gather_fp_finer(P, xtfi_[dim][side].recv, xtfi_[dim][side].send,
                            xtfi_[dim][side].sub, xtfi_[dim][side].n, dim, side, amr_RF);
            //Physical boundaries: bc_ib is empty for periodic and when the
            //all-same fast path is in use, so this is a no-op there.
            for(int ib : forest.face_groups[dim][side].bc_ib)
                apply_domain_bc_fp(block_fp_dbg(ib, dim), dim, side);
        }
    }

    void Exchange_fp(){
        if(nblocks<=1 && forest.max_level()==0) return;
        //SPD_EXCHANGE_CHECK=1 runs both exchange implementations from the
        //same pre-state and reports the first element where they disagree.
        //The forest path is verified bit-exact against a single block, so it
        //is the reference; this is also the A/B harness each step of the
        //exchange rework is checked with.
        if(exchange_check()){
            for(int dim=0; dim<3; dim++){
                if(!cfg.active[dim]) continue;
                SD_Solution& P = fp_pack(dim);
                SD_Vector pre("xchk_pre",  P.Vector.layout());
                SD_Vector packed("xchk_pk", P.Vector.layout());
                Kokkos::deep_copy(pre, P.Vector);
                gather_all_fp(dim);
                Kokkos::deep_copy(packed, P.Vector);
                Kokkos::deep_copy(P.Vector, pre);
                forest_exchange_fp(forest, blocks, dim);
                report_exchange_diff(P, packed, dim);
            }
            return;   //the forest result is left in place
        }
        //The transaction tables cover the uniform case too (one same-level
        //transaction per block face), so they are the single batched path.
        //
        //This used to take block_boundary_sd_b on a uniform forest. That kernel
        //is correct on the host and silently exchanges NOTHING under CUDA: with
        //SPD_EXCHANGE_CHECK it leaves 0 of 102400 entries changed where the
        //forest path changes 10240, and a decodable pattern shows every ghost
        //still holding its pre-exchange value. Every multiblock GPU run was
        //therefore missing its flux-point halo -- for all three fallback styles,
        //which is why "uniform forest == one block" held exactly on CPU and not
        //on GPU. The mechanism inside that kernel is not understood (captures,
        //loop bounds and destination indices all read back correct on device),
        //so it is removed rather than left as a default anyone can select.
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            if(no_pack()) forest_exchange_fp(forest, blocks, dim);
            else          gather_all_fp(dim);
        }
    }

    //The FV flux the cascade assembles into: hydro's F_*, MHD's level-0 F0_*.
    FV_Solution& fv_flux_pack(int dim){
        if constexpr (is_mhd)
            return dim==_x_ ? pv.F0_x : dim==_y_ ? pv.F0_y : pv.F0_z;
        else
            return dim==_x_ ? pv.F_x : dim==_y_ ? pv.F_y : pv.F_z;
    }

    //Conservative flux correction over the fine->coarse table. Hydro only for
    //now: MHD's level-0 flux arrays (F0_*) have no pack view, so it stays on
    //the per-block path.
    void correct_cf_fv_flux_batched(int dim){
        if(forest.max_level()==0 || !cfg.active[dim]) return;
        FV_Solution& C = fv_flux_pack(dim);
        //All blocks in a pack share extents, so one set of coarse dimensions
        //serves every transaction.
        const SD_Solution& S = pv.W_cv;
        const int nx=S.nx, ny=S.ny, nz=S.nz;
        const int Ncx=(S.Nx-2*NGHx)*nx, Ncy=(S.Ny-2*NGHy)*ny, Ncz=(S.Nz-2*NGHz)*nz;
        GHOST_LOCALS;
        const int lo = (dim==_x_?sghx:dim==_y_?sghy:sghz);
        const int hi = lo + (dim==_x_?Ncx:dim==_y_?Ncy:Ncz);
        for(int side=0; side<2; side++){
            const int cface = (side==0 ? lo : hi);
            const int fface = (side==0 ? hi : lo);
            correct_cf_fv_flux_b(C, xtfi_[dim][side].recv, xtfi_[dim][side].send,
                                 xtfi_[dim][side].sub, xtfi_[dim][side].n,
                                 dim, side, cface, fface,
                                 Ncx, Ncy, Ncz, nx, ny, nz);
        }
    }

    //SD/fp conservative flux correction over the fine->coarse table.
    void correct_cf_flux_batched(int dim){
        if(forest.max_level()==0 || !cfg.active[dim]) return;
        SD_Solution& F = dim==_x_ ? pv.F_ader_fp_x
                       : dim==_y_ ? pv.F_ader_fp_y : pv.F_ader_fp_z;
        //The per-block reference (correct_coarse_fine_flux) always restricts
        //with amr_RF. Selecting amr_RF_fp here on a transverse-count match was
        //a caller-side bug: the two have the same extent whenever the flux- and
        //solution-point counts coincide, so this silently picked the wrong
        //weights and the correction stopped being the reference's.
        Matrix R = amr_RF;
        for(int side=0; side<2; side++)
            correct_cf_flux_b(F, xtfi_[dim][side].recv, xtfi_[dim][side].send,
                              xtfi_[dim][side].sub, xtfi_[dim][side].n,
                              dim, side, R);
    }

    //Same dispatch as the FV counterpart. The batched kernel is verified
    //bit-identical to the forest path (dt trace over a 516-step 2-level AMR
    //run, plus every amr/smr config), and it removes the per-face scratch
    //allocation that fenced the device: on a 2-level KH run it took the launch
    //count from 974k to 486k, with restrict_face_overlap_sp (197k) gone and the
    //allocation traffic down 95-98%.
    void correct_cf_flux_dim(int dim){
        //MHD reaches this through the same block_Ffp arrays (F_ader_fp_*) and the
        //same amr_RF matrix as hydro, so the batched kernel applies unchanged --
        //it was only ever gated off because nothing had checked it.
        if(new_xchg() && (!is_mhd || mhd_batched())) correct_cf_flux_batched(dim);
        else correct_coarse_fine_flux(forest, blocks, dim);
    }

    //Whichever implementation is selected, for one direction.
    void correct_cf_fv_flux_dim(int dim){
        if(new_xchg() && (!is_mhd || mhd_batched())) correct_cf_fv_flux_batched(dim);
        else correct_coarse_fine_fv_flux(forest, blocks, dim);
    }

    //Same-level FV flux symmetrization over the side-0 same-level table.
    void symmetrize_fv_flux_batched(int dim){
        if(!cfg.active[dim]) return;
        FV_Solution& C = fv_flux_pack(dim);
        const SD_Solution& S = pv.W_cv;
        const int Ncx=(S.Nx-2*NGHx)*S.nx, Ncy=(S.Ny-2*NGHy)*S.ny, Ncz=(S.Nz-2*NGHz)*S.nz;
        GHOST_LOCALS;
        const int lo = (dim==_x_?sghx:dim==_y_?sghy:sghz);
        const int hi = lo + (dim==_x_?Ncx:dim==_y_?Ncy:Ncz);
        symmetrize_same_level_fv_flux_b(C, xt_[dim][0].recv, xt_[dim][0].send,
                                        xt_[dim][0].n, dim, lo, hi, Ncx, Ncy, Ncz);
    }

    //Whichever implementation is selected, for one direction.
    void symmetrize_fv_flux_dim(int dim){
        if(new_xchg() && (!is_mhd || mhd_batched())) symmetrize_fv_flux_batched(dim);
        else symmetrize_same_level_fv_flux(forest, blocks, dim);
    }

    //Single-valued FV fluxes at every block interface: the coarse side of a
    //level jump takes the restriction of the fine fluxes covering it, and a
    //same-level pair takes its shared average. spd's _enforce_flux_consistency.
    //
    //This has to run before the flux is *used*, not just before it is
    //committed. The cascade builds a candidate update from the assembled flux
    //and then runs the DMP test on that candidate to decide what to demote, so
    //a flux that is still double-valued at a block interface biases the very
    //test driving the cascade: cells demote on the mismatch rather than on the
    //solution, the next revision inherits a worse candidate, and dt collapses.
    void enforce_fv_flux_consistency(){
        if(forest.max_level()>0)
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) correct_cf_fv_flux_dim(dim);
        for(int dim=0; dim<3; dim++)
            if(cfg.active[dim]) symmetrize_fv_flux_dim(dim);
    }

    //All relations for one direction of an FV field, as batched gathers over
    //the same transaction tables the flux points use. Physical boundaries stay
    //a per-block call: there are few of them and they need no neighbour.
    void gather_all_fv(FV_Solution& P, FV_Solution Block::*member, int dim,
                       bool take_max){
        const int ngh = nGH_rt[dim];
        for(int side=0; side<2; side++){
            gather_fv_same(P, xt_[dim][side].recv, xt_[dim][side].send,
                           xt_[dim][side].n, dim, side, ngh);
            gather_fv_coarser(P, xtco_[dim][side].recv, xtco_[dim][side].send,
                              xtco_[dim][side].sub, xtco_[dim][side].n, dim, side, ngh);
            gather_fv_finer(P, xtfi_[dim][side].recv, xtfi_[dim][side].send,
                            xtfi_[dim][side].sub, xtfi_[dim][side].n, dim, side,
                            ngh, take_max);
            for(int ib : forest.face_groups[dim][side].bc_ib)
                apply_domain_bc_fv(blocks[ib].*member, dim, side, ngh);
        }
    }

    //SPD_EXCHANGE_CHECK=1 runs both FV implementations from the same pre-state
    //and reports where they disagree, exactly as Exchange_fp does. The forest
    //path is the reference.
    void Exchange_fv_check(FV_Solution& P, FV_Solution Block::*member, int dim,
                           bool take_max){
        FV_Vector pre("fvchk_pre", P.Vector.layout());
        FV_Vector packed("fvchk_pk", P.Vector.layout());
        Kokkos::deep_copy(pre, P.Vector);
        gather_all_fv(P, member, dim, take_max);
        Kokkos::deep_copy(packed, P.Vector);
        Kokkos::deep_copy(P.Vector, pre);
        if(take_max) forest_exchange_fv_max(forest, blocks, member, dim);
        else         forest_exchange_fv(forest, blocks, member, dim);
        report_fv_diff(P, packed, dim);
    }

    void Exchange_fv_field(FV_Solution Block::*member, FV_Solution* packed=nullptr){
        //SPD_NEW_XCHG routes the FV halo through the transaction tables. It
        //needs the whole-pack view, so a field without one still takes the
        //forest path.
        if(new_xchg() && packed){
            for(int dim=0; dim<3; dim++){
                if(!cfg.active[dim]) continue;
                if(exchange_check()) Exchange_fv_check(*packed, member, dim, false);
                else                 gather_all_fv(*packed, member, dim, false);
            }
            //Corner pass: after every direction, refill each transverse ghost
            //corner from the same-level neighbour that owns it. Batched, or it
            //undoes the saving -- per block this ran after every exchange.
            if(forest.max_level()>0)
                for(int dim=0; dim<3; dim++){
                    if(!cfg.active[dim]) continue;
                    const int ngh = nGH_rt[dim];
                    for(int side=0; side<2; side++)
                        gather_fv_same(*packed, xt_[dim][side].recv,
                                       xt_[dim][side].send, xt_[dim][side].n,
                                       dim, side, ngh);
                }
            return;
        }
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            if(forest.max_level()>0 || no_pack()){
                forest_exchange_fv(forest, blocks, member, dim);
            } else if(packed){
                block_boundary_fv_b(*packed,nbrL_[dim],nbrR_[dim],
                                    typL_[dim],typR_[dim],dim);
            } else {
                for(int b=0; b<nblocks; b++){
                    int L,R,tL,tR;
                    neighbors_uniform(b,dim,L,R,tL,tR);
                    block_boundary_fv(blocks[b].*member,blocks[L].*member,blocks[R].*member,tL,tR,dim);
                }
            }
        }
        if(forest.max_level()>0)
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim])
                    forest_exchange_fv_same(forest, blocks, member, dim);
    }

    //All relations of one SD field for one direction, as batched gathers over the
    //same transaction tables the flux points use. This is forest_exchange_sd's
    //work with the per-block loop removed: that loop was 45.5% of the fenced
    //mixed-level MHD advance, all of it in the face-B exchange, which runs twice
    //a step over every leaf.
    //
    //The semantics line up exactly, which is why the fp gathers can be reused
    //verbatim: forest_exchange_sd with cf_prolong=true does copy_face_to_ghost on
    //BOTH the coarser and the finer side, i.e. a pure ghost fill. (The
    //set_interface_flux variant that also writes the shared interior face belongs
    //to forest_sync_face_B, a different function -- do not conflate them.)
    //
    //Matrices come from prolong_mat_for / restrict_mat_for on the pack view, not
    //the bare amr_P / amr_RF that gather_all_fp passes: the per-block path selects
    //them per array, and face B is one of the arrays where the choice differs.
    void gather_all_sd(SD_Solution& P, SD_Solution Block::*member, int dim,
                       bool cf_prolong){
        (void)member;
        for(int side=0; side<2; side++){
            gather_fp_same(P, xt_[dim][side].recv, xt_[dim][side].send,
                           xt_[dim][side].n, dim, side);
            if(cf_prolong){
                gather_fp_coarser(P, xtco_[dim][side].recv, xtco_[dim][side].send,
                                  xtco_[dim][side].sub, xtco_[dim][side].n,
                                  dim, side, prolong_mat_for(P, dim));
                gather_fp_finer(P, xtfi_[dim][side].recv, xtfi_[dim][side].send,
                                xtfi_[dim][side].sub, xtfi_[dim][side].n,
                                dim, side, restrict_mat_for(P, dim));
            } else {
                //cf_prolong=false: the ghost takes the block's OWN face value.
                //Both relation lists get mirrored, exactly as forest_exchange_sd
                //does, and the receiver ids are the tables' recv arrays.
                mirror_faces_b(P, xtco_[dim][side].recv, xtco_[dim][side].n, dim, side);
                mirror_faces_b(P, xtfi_[dim][side].recv, xtfi_[dim][side].n, dim, side);
            }
            for(int ib : forest.face_groups[dim][side].bc_ib)
                apply_domain_bc_fp(blocks[ib].*member, dim, side);
        }
        //SPD_BREAK_SDGATHER=1 perturbs one entry, which validates the comparison
        //itself: unless this makes Exchange_sd_check report a difference, the check
        //is blind and its "agreement" means nothing. Two weaker controls were tried
        //first and BOTH passed for the wrong reason -- omitting the same-level
        //write is invisible because the exchange runs twice a step and the ghosts
        //already hold the right values, and gathering from the opposite side turns
        //out to be a no-op on an all-same face table.
        //Done with a 0-D deep_copy rather than a device lambda: nvcc cannot
        //generate its stub for a __device__ lambda inside a member function that
        //also takes a pointer-to-member parameter (it compiles on the host build
        //and fails only under CUDA, which is a bad way to find out).
        if(getenv("SPD_BREAK_SDGATHER")){
            auto sv = Kokkos::subview(P.Vector,0,0,0,0,0,0,0,0);
            double v = 0.0;
            Kokkos::deep_copy(v, sv);
            v += 1.0;
            Kokkos::deep_copy(sv, v);
        }
    }

    //NOT BATCHED, and here is why, because it looks like it should be.
    //
    //correct_coarse_fine_emf restricts the covering fine EMFs onto a coarse-shaped
    //buffer and set_interface_flux'es it onto both copies of the shared interface --
    //structurally identical to correct_coarse_fine_flux, whose batched twin
    //correct_cf_flux_b is generic in n_ader/n_var and takes the matrix as a
    //parameter. Reusing it for the EMF families is therefore the obvious move, and
    //it is WRONG: measured 640 differing entries with max|diff| = 1.0 against the
    //per-block path on a 2-level OT.
    //
    //The reason is the lattice. The flux arrays carry p+1 points per element in the
    //transverse direction; the edge-EMF arrays carry p+2 (edge points), and
    //restrict_mat_for picks amr_RF_fp for that count. correct_cf_flux_b's sub-face
    //index arithmetic (NB/2 halves in element units, nx points each) assumes the
    //flux lattice, so it addresses the wrong points on the edge lattice. A batched
    //EMF correction needs its own index mapping, not this one.
    //
    //It is also worth fixing the CORRECTNESS defect first: amr_RF_fp averages the
    //two fine halves' node j, which are different physical points, so this
    //correction does not telescope (see the cf_flux gate and [[spd-k-known-issues]]).
    //Batching an operation that is about to change shape is premature.
    //
    //SPD_EXCHANGE_CHECK=1: run both implementations from the same pre-state and
    //report the first element where they disagree, the forest path being the
    //reference. Same harness the FV and fp exchanges are checked with.
    void Exchange_sd_check(SD_Solution& P, SD_Solution Block::*member, int dim,
                           bool cf_prolong){
        SD_Vector pre("sdchk_pre", P.Vector.layout());
        SD_Vector packed("sdchk_pk", P.Vector.layout());
        Kokkos::deep_copy(pre, P.Vector);
        gather_all_sd(P, member, dim, cf_prolong);
        Kokkos::deep_copy(packed, P.Vector);
        Kokkos::deep_copy(P.Vector, pre);
        forest_exchange_sd(forest, blocks, member, dim, cf_prolong);
        //Direct max-diff, always printed: report_exchange_diff has a report cap
        //the fp exchange consumes first, so relying on it here reports nothing
        //whether the gather is right or wrong.
        {
            auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.Vector);
            auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), packed);
            double worst = 0.0; long nbad = 0;
            for(size_t i0=0;i0<a.extent(0);i0++)
            for(size_t i1=0;i1<a.extent(1);i1++)
            for(size_t i2=0;i2<a.extent(2);i2++)
            for(size_t i3=0;i3<a.extent(3);i3++)
            for(size_t i4=0;i4<a.extent(4);i4++)
            for(size_t i5=0;i5<a.extent(5);i5++)
            for(size_t i6=0;i6<a.extent(6);i6++)
            for(size_t i7=0;i7<a.extent(7);i7++){
                double d = fabs(a(i0,i1,i2,i3,i4,i5,i6,i7)-b(i0,i1,i2,i3,i4,i5,i6,i7));
                if(d>0){ nbad++; worst = worst>d?worst:d; }
            }
            if(Master)
                std::cout<<"[sdchk] dim="<<dim<<" step="<<this->n_step
                         <<"  differing="<<nbad<<"  max|forest-batched|="<<worst<<std::endl;
        }
    }

    void Exchange_sd_field(SD_Solution Block::*member, int dim,
                           bool cf_prolong=true, SD_Solution* packed=nullptr){
        if(!cfg.active[dim]) return;
        if(nblocks<=1 && forest.max_level()==0) return;
        if(forest.max_level()>0){
            //A field with a whole-pack view goes through the tables; one without
            //still takes the per-block forest path.
            //SPD_NO_SD_GATHER=1 forces the per-block forest path for SD fields
            //alone, so the batched gather can be A/B'd without also switching the
            //fp and FV exchanges the way SPD_OLD_XCHG does.
            static const bool no_sd_gather = getenv("SPD_NO_SD_GATHER")!=nullptr;
            if(new_xchg() && packed && !no_sd_gather){
                if(exchange_check()) Exchange_sd_check(*packed, member, dim, cf_prolong);
                else                 gather_all_sd(*packed, member, dim, cf_prolong);
                return;
            }
            forest_exchange_sd(forest, blocks, member, dim, cf_prolong);
            return;
        }
        for(int b=0; b<nblocks; b++){
            int L,R,tL,tR;
            neighbors_uniform(b,dim,L,R,tL,tR);
            block_boundary_sd(blocks[b].*member, blocks[L].*member, blocks[R].*member,
                              tL, tR, dim);
        }
    }

    void Exchange_fv_field_max(FV_Solution Block::*member, FV_Solution* packed=nullptr){
        //The cascade halo runs once per revision, so this was the single
        //largest source of launches in an AMR cascade run (41% of them).
        if(new_xchg() && packed){
            for(int dim=0; dim<3; dim++){
                if(!cfg.active[dim]) continue;
                if(exchange_check()) Exchange_fv_check(*packed, member, dim, true);
                else                 gather_all_fv(*packed, member, dim, true);
            }
            if(forest.max_level()>0)
                for(int dim=0; dim<3; dim++){
                    if(!cfg.active[dim]) continue;
                    const int ngh = nGH_rt[dim];
                    for(int side=0; side<2; side++)
                        gather_fv_same(*packed, xt_[dim][side].recv,
                                       xt_[dim][side].send, xt_[dim][side].n,
                                       dim, side, ngh);
                }
            return;
        }
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            if(forest.max_level()>0){
                forest_exchange_fv_max(forest, blocks, member, dim);
            } else {
                for(int b=0; b<nblocks; b++){
                    int L,R,tL,tR;
                    neighbors_uniform(b,dim,L,R,tL,tR);
                    block_boundary_fv(blocks[b].*member,blocks[L].*member,blocks[R].*member,tL,tR,dim);
                }
            }
        }
        if(forest.max_level()>0)
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim])
                    forest_exchange_fv_same(forest, blocks, member, dim);
    }

    //Batched phases: one kernel per direction over every block of the pack,
    //replacing a per-block loop of identical kernels. Same arithmetic per
    //cell, so results are unchanged; only the launch count differs.
    void Compute_Fluxes_batched(){
        if(cfg.active[_x_]) compute_fluxes(pv.U_ader_fp_x,pv.F_ader_fp_x,_vx_,_vy_,_vz_);
        if(cfg.active[_y_]) compute_fluxes(pv.U_ader_fp_y,pv.F_ader_fp_y,_vy_,_vz_,_vx_);
        if(cfg.active[_z_]) compute_fluxes(pv.U_ader_fp_z,pv.F_ader_fp_z,_vz_,_vx_,_vy_);
    }

    void Riemann_Solver_batched(){
        bool visc = blocks[0].viscosity;
        if(cfg.active[_x_])
            sd_riemann_solver(pv.U_ader_fp_x,pv.F_ader_fp_x,_vx_,_vy_,_vz_,_x_,visc);
        if(cfg.active[_y_])
            sd_riemann_solver(pv.U_ader_fp_y,pv.F_ader_fp_y,_vy_,_vz_,_vx_,_y_,visc);
        if(cfg.active[_z_])
            sd_riemann_solver(pv.U_ader_fp_z,pv.F_ader_fp_z,_vz_,_vx_,_vy_,_z_,visc);
    }

    void Interpolate_to_fp_batched(){
        Matrix sp_to_fp = blocks[0].sp_to_fp;
        if(cfg.active[_x_])
            transform_a_to_b_1d(pv.U_ader_sp,pv.U_ader_fp_x,sp_to_fp,_x_);
        if(cfg.active[_y_])
            transform_a_to_b_1d(pv.U_ader_sp,pv.U_ader_fp_y,sp_to_fp,_y_);
        if(cfg.active[_z_])
            transform_a_to_b_1d(pv.U_ader_sp,pv.U_ader_fp_z,sp_to_fp,_z_);
    }

    //Packed W_sp -> W_cv, the same sweep the blocks do individually
    void transform_sp_to_cv_batched(SD_Solution src, SD_Solution dst){
        transform_a_to_b(src, dst, pv.T_sweep, blocks[0].sp_to_cv);
    }

    void Solve_fluxes_hydro(){
        { Region r("Fluxes_pre");
          Interpolate_to_fp_batched();
          Compute_Fluxes_batched(); }
        { Region r("Exchange_fp"); Exchange_fp(); }
        { Region r("Riemann_Solver"); Riemann_Solver_batched(); }
        if(forest.max_level()>0){
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) correct_cf_flux_dim(dim);
        }
        //Viscosity is opt-in at runtime (athenak-style): hydro/nu>0 in the
        //input file sets Hydro_ader::viscosity. The second flux-point exchange
        //is needed because the viscous flux depends on gradients that the
        //first exchange has only just made available.
        if(blocks[0].viscosity){
            Region r("Viscosity");
            for(int b=0;b<nblocks;b++) blocks[b].Viscosity(Xd[b].h,Yd[b].h,Zd[b].h);
            Exchange_fp();
            for(int b=0;b<nblocks;b++) blocks[b].Rusanov_Solver();
        }
    }

    //U_sp -> U_cv over the whole pack (the sweep FV_begin does per block)
    void FV_begin_batched(){
        transform_a_to_b(pv.U_sp, pv.U_cv, pv.T_sweep, blocks[0].sp_to_cv);
    }

    void FV_end_batched(){
        transform_a_to_b(pv.U_cv, pv.U_sp, pv.T_sweep, blocks[0].cv_to_sp);
    }

    //Pure MUSCL pins the blend to 1 everywhere, so this is one fill over the
    //whole pack rather than one per block.
    //Primitives over the whole pack; the detection stencil (not yet batched)
    //is skipped entirely under pure MUSCL, which never reads the flags.
    void FV_detect_batched(){
        compute_primitives(pv.U_old, pv.W_old);
        if(cfg.fv_only) return;
        compute_primitives(pv.U_new, pv.W_new);
        detect_pack();
    }

    //Detection over the whole pack: one launch per criterion. The per-block
    //loop this replaces was the dominant cost once AMR started using the
    //cascade, which calls detection once per revision.
    void detect_pack(){
        detect_troubles_b(pv.W_new, pv.W_old, pv.troubles, pv.flagged,
                          pv.alpha_x, pv.alpha_y, pv.alpha_z,
                          fvxc_p, fvx_p, fvyc_p, fvy_p, fvzc_p, fvz_p,
                          Xd[0].p, 1, (1<<_d_)|(1<<_p_));
    }

    //SD face fluxes -> FV faces -> candidate update, over the whole pack.
    //The per-block FV face coordinates come in as the geometry pack, so one
    //launch spans blocks that sit at different refinement levels.
    void FV_flux_update_batched(int ader){
        if(cfg.active[_x_])
            face_integral_b(pv.F_ader_fp_x, pv.F_x, pv.T_fp_x,
                            blocks[0].sp_to_cv, ader, _x_);
        if(cfg.active[_y_])
            face_integral_b(pv.F_ader_fp_y, pv.F_y, pv.T_fp_y,
                            blocks[0].sp_to_cv, ader, _y_);
        if(cfg.active[_z_])
            face_integral_b(pv.F_ader_fp_z, pv.F_z, pv.T_fp_z,
                            blocks[0].sp_to_cv, ader, _z_);
        fv_update_solution_b(pv.U_new, pv.U_old, pv.U_cv,
            pv.F_x, fvx_p, pv.F_y, fvy_p, pv.F_z, fvz_p,
            blocks[0].wt, ader, dt, 0);
    }

    void FV_commit_batched(int ader){
        fv_update_solution_b(pv.U_new, pv.U_old, pv.U_cv,
            pv.F_x, fvx_p, pv.F_y, fvy_p, pv.F_z, fvz_p,
            blocks[0].wt, ader, dt, 1);
    }

    //Candidate update: same as the commit but it leaves U_cv alone.
    void FV_candidate_batched(int ader){
        fv_update_solution_b(pv.U_new, pv.U_old, pv.U_cv,
            pv.F_x, fvx_p, pv.F_y, fvy_p, pv.F_z, fvz_p,
            blocks[0].wt, ader, dt, 0);
    }

    //Both lower cascade levels' fluxes, over the whole pack: two launches per
    //direction per stage instead of two per direction per block.
    void cascade_levels_pack(int ader){
        level_fluxes_b(pv.W_old,
            fvxc_p, fvx_p, pv.F1_x,
            fvyc_p, fvy_p, pv.F1_y,
            fvzc_p, fvz_p, pv.F1_z,
            ader, blocks[0].wt, dt, true);
        level_fluxes_b(pv.W_old,
            fvxc_p, fvx_p, pv.F2_x,
            fvyc_p, fvy_p, pv.F2_y,
            fvzc_p, fvz_p, pv.F2_z,
            ader, blocks[0].wt, dt, false);
        Kokkos::deep_copy(pv.cascade.Vector, 0.0);
    }

    //Pick each face's flux from the pooled cascade level, over the whole pack.
    //This sits inside the revision loop, so per-block it was the launch that
    //repeated most: three directions x max_revs x block count, every stage.
    void cascade_assemble_pack(){
        assign_face_flux_b(pv.F_x, pv.F1_x, pv.F2_x, pv.cascade, _x_);
        if(cfg.active[_y_]) assign_face_flux_b(pv.F_y, pv.F1_y, pv.F2_y, pv.cascade, _y_);
        if(cfg.active[_z_]) assign_face_flux_b(pv.F_z, pv.F1_z, pv.F2_z, pv.cascade, _z_);
    }

    //Blend the fallback flux into every face over the whole pack. This was the
    //last per-block host loop in the blend update; once the halo exchanges were
    //batched it dominated what remained (447,840 of 605,961 launches on the
    //fig-21 MUSCL profile).
    void FV_blend_batched(int ader){
        fallback_fluxes_b(pv.W_old, pv.theta,
            fvxc_p, fvx_p, pv.F_x,
            fvyc_p, fvy_p, pv.F_y,
            fvzc_p, fvz_p, pv.F_z,
            ader, blocks[0].wt, dt);
    }

    void FV_theta_batched(){
        if(cfg.fv_only){
            Kokkos::deep_copy(pv.theta.Vector, 1.0);
            return;
        }
        if(cfg.fv_only){ Kokkos::deep_copy(pv.theta.Vector, 1.0); return; }
        if(cfg.blending){
            apply_blending_b(pv.flagged, pv.theta_tmp);
            blending_ring_b(pv.theta_tmp, pv.theta);
        }
        else
            theta_from_flagged_b(pv.flagged, pv.theta);
    }

    void FV_Update_solution_hydro(){
        { Region r("FV_begin"); FV_begin_batched(); }
        for(int ader=0;ader<n_ader;ader++){
            { Region r("FV_flux_update"); FV_flux_update_batched(ader); }
            { Region r("Exchange_U_old"); Exchange_fv_field(&Block::U_old,&pv.U_old); }
            { Region r("Exchange_U_new"); Exchange_fv_field(&Block::U_new,&pv.U_new); }
            { Region r("FV_detect"); FV_detect_batched(); }
            if(!cfg.fv_only){ Region r("Exchange_flagged");
                                 Exchange_fv_field(&Block::flagged,&pv.flagged); }
            { Region r("FV_theta"); FV_theta_batched(); }
            if(!cfg.fv_only){ Region r("Exchange_theta");
                                 Exchange_fv_field(&Block::theta,&pv.theta); }
            { Region r("FV_blend"); FV_blend_batched(ader); }
            if(forest.max_level()>0){
                Region r("correct_cf_fv_flux");
                for(int dim=0; dim<3; dim++)
                    if(cfg.active[dim])
                        correct_cf_fv_flux_dim(dim);
            }
            { Region r("FV_commit"); FV_commit_batched(ader); }
        }
        { Region r("FV_end"); FV_end_batched(); }
    }

    //Mesh-level MOOD cascade, the hydro counterpart of MHD_MOOD_update. The
    //level fluxes are built once per ader step; a revision then re-assembles,
    //re-tests, and stops as soon as no cell demotes. The cascade halo takes
    //the max, so a demotion on either side of a level jump is seen by both
    //(Exchange_fv_field_max), which is what keeps a coarse-fine face
    //single-valued.
    void FV_Update_solution_hydro_cascade(){
        { Region r("FV_begin"); FV_begin_batched(); }
        for(int ader=0;ader<n_ader;ader++){
            { Region r("FV_flux_update"); FV_flux_update_batched(ader); }
            { Region r("Exchange_U_old"); Exchange_fv_field(&Block::U_old,&pv.U_old); }
            { Region r("FV_cascade_levels");
              compute_primitives(pv.U_old, pv.W_old);
              cascade_levels_pack(ader); }
            for(int rev=0; rev<cfg.max_revs; rev++){
                { Region r("FV_cascade_candidate");
                  cascade_assemble_pack();
                  //The candidate the DMP test judges must come from a
                  //single-valued flux, or the test reads the interface
                  //mismatch as trouble. Idempotent, so re-running it every
                  //revision is safe: the cascade level only ever rises, and
                  //assign_face_cell leaves untouched faces alone.
                  enforce_fv_flux_consistency();
                  FV_candidate_batched(ader); }
                //SED limits against a two-cell stencil of the candidate, so
                //each revision needs its own U_new halo.
                { Region r("Exchange_U_new"); Exchange_fv_field(&Block::U_new,&pv.U_new); }
                int demoted = 0;
                { Region r("FV_cascade_detect");
                  compute_primitives(pv.U_new, pv.W_new);
                  detect_pack();
                  demoted = update_cascade_b(pv.flagged, pv.cascade, 2); }
                #ifdef MPI
                int g; MPI_Allreduce(&demoted,&g,1,MPI_INT,MPI_SUM,Comm); demoted=g;
                #endif
                if(demoted==0) break;
                { Region r("Exchange_cascade"); Exchange_fv_field_max(&Block::cascade,&pv.cascade); }
            }
            { Region r("FV_cascade_assemble"); cascade_assemble_pack(); }
            { Region r("enforce_fv_flux_consistency"); enforce_fv_flux_consistency(); }
            { Region r("FV_commit"); FV_commit_batched(ader); }
        }
        { Region r("FV_end"); FV_end_batched(); }
    }

    void Update_solution_hydro(){
        if(cfg.fallback){
            if(cfg.mood_cascade) FV_Update_solution_hydro_cascade();
            else                 FV_Update_solution_hydro();
        }
        else for(int b=0;b<nblocks;b++)
            blocks[b].Update_solution(Xd[b].h,Yd[b].h,Zd[b].h);
    }

    void Exchange_E_mhd(){
        if constexpr (!is_mhd) return;
        bool az = cfg.active[_z_];
        //Edge EMF: CF ghosts mirrored (fp P/R unstable); same-level copied.
        Exchange_sd_field(&Block::Ez_ep_xy, _x_, false, &pv.Ez_ep_xy);
        Exchange_sd_field(&Block::Ez_ep_xy, _y_, false, &pv.Ez_ep_xy);
        if(az){
            Exchange_sd_field(&Block::Ey_ep_zx, _x_, false, &pv.Ey_ep_zx);
            Exchange_sd_field(&Block::Ey_ep_zx, _z_, false, &pv.Ey_ep_zx);
            Exchange_sd_field(&Block::Ex_ep_yz, _y_, false, &pv.Ex_ep_yz);
            Exchange_sd_field(&Block::Ex_ep_yz, _z_, false, &pv.Ex_ep_yz);
        }
    }

    //Shared face-B identity: left-neighbour value wins at each block interface.
    //Mixed-level: only same-level identity (CF interiors are owned by CT+EMF
    //correction). Use Exchange_face_B_mhd for CF/same ghost fill.
    void Sync_face_B_mhd(){
        if constexpr (!is_mhd) return;
        if(nblocks<=1 && forest.max_level()==0) return;
        auto sync_same = [&](SD_Solution Block::*member, int dim){
            if(!cfg.active[dim]) return;
            if(forest.max_level()==0){
                for(int b=0; b<nblocks; b++){
                    int L,R,tL,tR;
                    neighbors_uniform(b,dim,L,R,tL,tR);
                    sync_shared_face_sd(blocks[b].*member, blocks[L].*member,
                                        blocks[R].*member, tL, tR, dim);
                }
                return;
            }
            const FaceGroups& g = forest.face_groups[dim][0];
            for(size_t k=0; k<g.same_ib.size(); k++)
                sync_shared_face_sd(blocks[g.same_ib[k]].*member,
                                    blocks[g.same_jb[k]].*member,
                                    blocks[g.same_ib[k]].*member,
                                    _periodic_, _periodic_, dim);
        };
        sync_same(&Block::Bx_fp_x, _x_);
        sync_same(&Block::By_fp_y, _y_);
        if(cfg.active[_z_]) sync_same(&Block::Bz_fp_z, _z_);
    }

    //Ghost fill for face-staggered B (same-level copy + CF prolong/restrict).
    //Writes only ghost faces — does not overwrite CT-owned interior faces.
    void Exchange_face_B_mhd(){
        if constexpr (!is_mhd) return;
        Exchange_sd_field(&Block::Bx_fp_x, _x_, true, &pv.Bx_fp_x);
        Exchange_sd_field(&Block::By_fp_y, _y_, true, &pv.By_fp_y);
        if(cfg.active[_z_])
            Exchange_sd_field(&Block::Bz_fp_z, _z_, true, &pv.Bz_fp_z);
    }

    //Global NAD scales for mood_nad_scale=grange|gcfl. The reduction is over the
    //WHOLE mesh: reduced per block it gives every block its own band, so the same
    //flow is judged differently either side of a block face (measured on OT: the
    //2x2 multiblock diverged from single-block under gcfl and agreed exactly under
    //the local `relative` band). dxmin is the global minimum too -- under AMR the
    //gcfl softening dt*vmax/dxmin would otherwise differ level by level.
    void mhd_reduce_nad_gscales(){
        if constexpr (!is_mhd) return;
        if(cfg.mood_nad_scale!=_nad_scale_grange_ && cfg.mood_nad_scale!=_nad_scale_gcfl_)
            return;
        double gmin[8], gmax[8], gs[8], vmax=0.0;
        for(int v=0;v<8;v++){ gmin[v]=1e300; gmax[v]=-1e300; gs[v]=0.0; }
        for(int b=0;b<nblocks;b++) blocks[b].mood_partial_gscales(gmin,gmax,vmax);
        double dxmin=1e300;
        for(int b=0;b<nblocks;b++){
            if(cfg.active[_x_]) dxmin=std::min(dxmin,Xd[b].h);
            if(cfg.active[_y_]) dxmin=std::min(dxmin,Yd[b].h);
            if(cfg.active[_z_]) dxmin=std::min(dxmin,Zd[b].h);
        }
        #ifdef MPI
        double t[8];
        MPI_Allreduce(gmin,t,8,MPI_DOUBLE,MPI_MIN,Comm); for(int v=0;v<8;v++) gmin[v]=t[v];
        MPI_Allreduce(gmax,t,8,MPI_DOUBLE,MPI_MAX,Comm); for(int v=0;v<8;v++) gmax[v]=t[v];
        double r;
        MPI_Allreduce(&vmax,&r,1,MPI_DOUBLE,MPI_MAX,Comm); vmax=r;
        MPI_Allreduce(&dxmin,&r,1,MPI_DOUBLE,MPI_MIN,Comm); dxmin=r;
        #endif
        int nd = nblocks>0 ? blocks[0].n_det : 0;
        mhd_nad_finalize_gscales(gmin,gmax,vmax,nd,gs,this->dt,dxmin);
        for(int b=0;b<nblocks;b++) blocks[b].mood_set_gscales(gs);
    }

    //Flux AND edge-EMF consistency for the assembled cascade set. The flux half
    //reuses enforce_fv_flux_consistency; the EMF half has no counterpart
    //elsewhere, because the SD correct_coarse_fine_emf works on the SD edge
    //arrays that MOOD rebuilds on the FV lattice and then never corrected.
    //Without it the assembled EMF is multi-valued at a level jump and the CT
    //update stops being divergence-free there.
    void enforce_fv_emf_consistency(){
        if constexpr (!is_mhd) return;
        if(forest.max_level()==0) return;
        //Diagnostic A/B, same spirit as SPD_NO_PACK / SPD_NO_DEREFINE: skipping
        //this is what the cascade did before the correction existed.
        static const bool off = getenv("SPD_NO_FV_EMF")!=nullptr;
        if(off) return;
        for(int dim=0; dim<3; dim++)
            if(cfg.active[dim]) correct_coarse_fine_fv_emf(forest, blocks, dim);
        //After BOTH directions, because what this spreads into the diagonal
        //block is the value the face pass wrote. SPD_NO_EMF_CORNER=1 is the A/B
        //that decides whether a change in the gate came from here.
        if(!no_emf_corner()) spread_fv_emf_corners_b(pv.E0z, ect_.t, ect_.n);
    }

    //Batched MHD phases. The per-block loops they replace were the whole cost of
    //MHD under AMR: measured on a uniform 256^2 mesh at fixed total work, going
    //from 4 blocks to 256 blocks cost 7.20 s -> 79.94 s per 200 steps on an A100
    //(11.1x) while the same sweep on CPU was flat, so it is pure launch dispatch.
    //SPD_NO_MHD_BATCH=1 restores the per-block path; the two must agree bitwise.
    //Phase bits: 1 begin, 2 after_U_halo, 4 assemble, 8 commit, 16 Fluxes_pre,
    //32 Riemann_Solver, 64 B_to_U, 128 Compute_E, 256 E_Riemann, 512 the RK
    //bookkeeping tasks (save_state, copy_cons, cons_to_prim, combine, B_to_U,
    //compute_dt), 1024 mood/detect. SPD_MHD_BATCH_MASK selects which are batched,
    //which is how a mismatch against the per-block path gets bisected to one
    //phase instead of guessed at.
    static bool mhd_batched(int phase = 15){
        static const bool off = getenv("SPD_NO_MHD_BATCH") != nullptr;
        if(off) return false;
        static const int mask = getenv("SPD_MHD_BATCH_MASK")
                                ? atoi(getenv("SPD_MHD_BATCH_MASK")) : 2047;
        return (mask & phase) != 0;
    }

    //mood_begin over the whole pack. Same order of operations as
    //MHD_ader::mood_begin, one launch per step instead of one per block.
    void MOOD_begin_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        auto& b0 = blocks[0];
        transform_a_to_b(pv.U_sp, pv.U_cv, pv.T_sweep, b0.sp_to_cv);
        //mood_reset_face_B
        transform_a_to_b_2d_b(pv.Bx_fp_x, pv.Bxf, pv.TB_x, b0.sp_to_cv, _x_);
        transform_a_to_b_2d_b(pv.By_fp_y, pv.Byf, pv.TB_y, b0.sp_to_cv, _y_);
        if(az) transform_a_to_b_2d_b(pv.Bz_fp_z, pv.Bzf, pv.TB_z, b0.sp_to_cv, _z_);
        compute_B_cv_from_cf_b(pv.B_old_cv, pv.Bxf, pv.Byf, pv.Bzf, b0.fp_to_cv);
        //Scratch: the MHD per-block path hands face_integral its own U_ader_fp_*,
        //NOT the hydro T_fp_* arrays -- MHD never allocates those, so passing
        //pv.T_fp_* here handed the kernel an empty view. It went unnoticed in 2D
        //because face_integral_b only touches the scratch when BOTH transverse
        //directions are active; with z on it diverged from the per-block path.
        if(cfg.active[_x_])
            face_integral_b(pv.F_ader_fp_x, pv.F0_x, pv.U_ader_fp_x, b0.sp_to_cv, 0, _x_);
        if(cfg.active[_y_])
            face_integral_b(pv.F_ader_fp_y, pv.F0_y, pv.U_ader_fp_y, b0.sp_to_cv, 0, _y_);
        if(az)
            face_integral_b(pv.F_ader_fp_z, pv.F0_z, pv.U_ader_fp_z, b0.sp_to_cv, 0, _z_);
        if(az){
            edge_integral_b(pv.Ex_ep_yz, pv.E0x, b0.sp_to_cv, 0, _x_);
            edge_integral_b(pv.Ey_ep_zx, pv.E0y, b0.sp_to_cv, 0, _y_);
        }
        edge_integral_b(pv.Ez_ep_xy, pv.E0z, b0.sp_to_cv, 0, _z_);
        fv_update_solution_b(pv.U_new_fv, pv.U_old_fv, pv.U_cv,
                             pv.F0_x, fvx_p, pv.F0_y, fvy_p, pv.F0_z, fvz_p,
                             b0.wt, 0, dt, 0);
        mhd_set_candidate_B_b(pv.U_old_fv, pv.B_old_cv);
        }
    }

    //mood_ct_update + mood_commit_assembled over the whole pack, in the same
    //order MHD_ader::mood_commit_assembled uses.
    void MOOD_commit_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        auto& b0 = blocks[0];
        //mood_fluid_update(true)
        fv_update_solution_b(pv.U_new_fv, pv.U_old_fv, pv.U_cv,
                             pv.F0_x, fvx_p, pv.F0_y, fvy_p, pv.F0_z, fvz_p,
                             b0.wt, 0, dt, 1);
        //mood_ct_update
        transform_a_to_b_2d_b(pv.Bx_fp_x, pv.Bxf, pv.TB_x, b0.sp_to_cv, _x_);
        transform_a_to_b_2d_b(pv.By_fp_y, pv.Byf, pv.TB_y, b0.sp_to_cv, _y_);
        if(az) transform_a_to_b_2d_b(pv.Bz_fp_z, pv.Bzf, pv.TB_z, b0.sp_to_cv, _z_);
        fv_update_B_solution_b(pv.Bx_new, pv.Bx_old, pv.Bxf, pv.E0y, pv.E0z,
                               fvy_p, fvz_p, b0.wt, dt, 0, _x_, 1);
        fv_update_B_solution_b(pv.By_new, pv.By_old, pv.Byf, pv.E0z, pv.E0x,
                               fvz_p, fvx_p, b0.wt, dt, 0, _y_, 1);
        if(az)
            fv_update_B_solution_b(pv.Bz_new, pv.Bz_old, pv.Bzf, pv.E0x, pv.E0y,
                                   fvx_p, fvy_p, b0.wt, dt, 0, _z_, 1);
        compute_B_cv_from_cf_b(pv.B_new_cv, pv.Bxf, pv.Byf, pv.Bzf, b0.fp_to_cv);
        if(cfg.floor_cons) mhd_floor_cv_b(pv.U_cv, pv.B_new_cv);
        transform_a_to_b(pv.U_cv, pv.U_sp, pv.T_sweep, b0.cv_to_sp);
        transform_a_to_b_2d_b(pv.Bxf, pv.Bx_fp_x, pv.TB_x, b0.cv_to_sp, _x_);
        transform_a_to_b_2d_b(pv.Byf, pv.By_fp_y, pv.TB_y, b0.cv_to_sp, _y_);
        if(az) transform_a_to_b_2d_b(pv.Bzf, pv.Bz_fp_z, pv.TB_z, b0.cv_to_sp, _z_);
        }
    }

    //mood_detect over the whole pack, in MHD_ader::mood_detect's order.
    //
    //THIS WAS THE LARGEST REMAINING PER-BLOCK LOOP IN THE CODE, and every profile
    //in the project missed it because they all pinned mhd/mood_force_level=1,
    //where mood_detect() returns immediately and reads 0.000 s. With detection
    //live (mood_force_level=-1, the default) on a 352-leaf mesh over 200 steps it
    //was 21.280 s of a 24.471 s fenced total -- 87% -- against 1.683 s for
    //cf/correct_cf_emf. Per block per revision it ran ~13 kernels AND ended in a
    //device->host reduction for the demoted count, so 654 calls x 352 blocks came
    //to ~230k host syncs. Hydro has had FV_detect_batched since the hydro
    //batching went in; this is its MHD counterpart.
    //
    //The demoted count is one reduction over the pack (update_cascade_b), not one
    //per block, which is what removes the syncs.
    int MOOD_detect_batched(){
        if constexpr (!is_mhd) return 0;
        else {
        //MHD_ader::mood_detect opens with exactly this guard, and the batched
        //form MUST mirror it: at a pinned cascade level there is nothing to
        //detect, and running the detection anyway writes U_new_fv, B_new_cv,
        //troubles and cascade where the per-block path wrote nothing -- which
        //changes the assembled flux. Missing it made the two paths differ on the
        //p=0 dynamic AMR lane while every lane I had checked (all with detection
        //LIVE) agreed. The mirror image of the trap in CLAUDE.md rule 2: check
        //the feature both ON and OFF.
        if(cfg.mood_force_level>=0) return 0;
        const bool az = cfg.active[_z_];
        auto& b0 = blocks[0];
        //mood_fluid_update(false)
        fv_update_solution_b(pv.U_new_fv, pv.U_old_fv, pv.U_cv,
                             pv.F0_x, fvx_p, pv.F0_y, fvy_p, pv.F0_z, fvz_p,
                             b0.wt, 0, dt, 0);
        //mood_ct_update: mood_reset_face_B, then the three face updates and the
        //cell-centred field, exactly as MOOD_commit_batched does for the commit.
        transform_a_to_b_2d_b(pv.Bx_fp_x, pv.Bxf, pv.TB_x, b0.sp_to_cv, _x_);
        transform_a_to_b_2d_b(pv.By_fp_y, pv.Byf, pv.TB_y, b0.sp_to_cv, _y_);
        if(az) transform_a_to_b_2d_b(pv.Bz_fp_z, pv.Bzf, pv.TB_z, b0.sp_to_cv, _z_);
        fv_update_B_solution_b(pv.Bx_new, pv.Bx_old, pv.Bxf, pv.E0y, pv.E0z,
                               fvy_p, fvz_p, b0.wt, dt, 0, _x_, 1);
        fv_update_B_solution_b(pv.By_new, pv.By_old, pv.Byf, pv.E0z, pv.E0x,
                               fvz_p, fvx_p, b0.wt, dt, 0, _y_, 1);
        if(az)
            fv_update_B_solution_b(pv.Bz_new, pv.Bz_old, pv.Bzf, pv.E0x, pv.E0y,
                                   fvx_p, fvy_p, b0.wt, dt, 0, _z_, 1);
        compute_B_cv_from_cf_b(pv.B_new_cv, pv.Bxf, pv.Byf, pv.Bzf, b0.fp_to_cv);
        mhd_set_candidate_B_b(pv.U_new_fv, pv.B_new_cv);
        const int nd = mhd_detection_vars_b(pv.U_new_fv, pv.det_new);
        //One mesh-wide gscale array (Mesh::mhd_reduce_nad_gscales pushes the same
        //values into every block), so block 0's copy speaks for the pack.
        mhd_NAD_b(pv.det_new, pv.det_old, pv.troubles, cfg.nad_tolerance,
                  nd, b0.nad_gscale);
        mhd_PAD_b(pv.U_new_fv, pv.troubles);
        return update_cascade_b(pv.troubles, pv.mhd_cascade, 2);
        }
    }

    //mood_assemble over the whole pack: the same in-place selection into the
    //level-0 arrays, one launch per family instead of one per block.
    void MOOD_assemble_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        assign_face_flux_b(pv.F0_x, pv.F1m_x, pv.F2m_x, pv.mhd_cascade, _x_);
        assign_face_flux_b(pv.F0_y, pv.F1m_y, pv.F2m_y, pv.mhd_cascade, _y_);
        if(az) assign_face_flux_b(pv.F0_z, pv.F1m_z, pv.F2m_z, pv.mhd_cascade, _z_);
        if(az){
            mhd_assign_edge_E_b(pv.E0x, pv.E1x, pv.E2x, pv.mhd_cascade, _x_);
            mhd_assign_edge_E_b(pv.E0y, pv.E1y, pv.E2y, pv.mhd_cascade, _y_);
        }
        mhd_assign_edge_E_b(pv.E0z, pv.E1z, pv.E2z, pv.mhd_cascade, _z_);
        }
    }

    //mood_after_U_halo over the whole pack: the primitives, the FV-face copy of
    //the stage field, the detection variables, and both cascade levels of the face
    //flux and the corner EMF. This was 33.8% of the fenced advance at 256 blocks,
    //the single largest per-block dispatch cost in the MHD path.
    //
    //The NAD global scales stay on the mesh-wide reduction path
    //(mhd_reduce_nad_gscales); nothing here touches them.
    void MOOD_after_U_halo_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        mhd_compute_primitives_b(pv.U_old_fv, pv.W_fv);
        mhd_face_B_to_fv_b(pv.Bxf, pv.Bx_old, _x_);
        mhd_face_B_to_fv_b(pv.Byf, pv.By_old, _y_);
        if(az) mhd_face_B_to_fv_b(pv.Bzf, pv.Bz_old, _z_);
        const int nd = mhd_detection_vars_b(pv.U_old_fv, pv.det_old);
        for(int b=0;b<nblocks;b++) blocks[b].n_det = nd;
        for(int dim=0; dim<3; dim++){
            if(cfg.active[dim]){
                FV_Solution& F1 = (dim==_x_?pv.F1m_x:(dim==_y_?pv.F1m_y:pv.F1m_z));
                FV_Solution& F2 = (dim==_x_?pv.F2m_x:(dim==_y_?pv.F2m_y:pv.F2m_z));
                FV_Solution& Bn = (dim==_x_?pv.Bx_old:(dim==_y_?pv.By_old:pv.Bz_old));
                FV_Solution& U1 = (dim==_x_?pv.UCT1_x:(dim==_y_?pv.UCT1_y:pv.UCT1_z));
                FV_Solution& U2 = (dim==_x_?pv.UCT2_x:(dim==_y_?pv.UCT2_y:pv.UCT2_z));
                mhd_fv_fluxes_b(pv.W_fv,F1,Bn,U1, fvxc_p,fvx_p,fvyc_p,fvy_p,
                                fvzc_p,fvz_p, dim, true);
                mhd_fv_fluxes_b(pv.W_fv,F2,Bn,U2, fvxc_p,fvx_p,fvyc_p,fvy_p,
                                fvzc_p,fvz_p, dim, false);
            }
            const int d1=(dim==_z_?_x_:(dim==_y_?_z_:_y_));
            const int d2=(dim==_z_?_y_:(dim==_y_?_x_:_z_));
            if(cfg.active[d1] && cfg.active[d2]){
                FV_Solution& E1 = (dim==_x_?pv.E1x:(dim==_y_?pv.E1y:pv.E1z));
                FV_Solution& E2 = (dim==_x_?pv.E2x:(dim==_y_?pv.E2y:pv.E2z));
                mhd_four_state_E_b(E1,pv.W_fv, fvxc_p,fvx_p,fvyc_p,fvy_p,
                                   fvzc_p,fvz_p, dim, true);
                mhd_four_state_E_b(E2,pv.W_fv, fvxc_p,fvx_p,fvyc_p,fvy_p,
                                   fvzc_p,fvz_p, dim, false);
            }
        }
        //One fill over the whole pack instead of one per block; deep_copy covers
        //the ghosts, so a forced level needs no halo (as in MHD_ader).
        Kokkos::deep_copy(pv.mhd_cascade.Vector,
                          cfg.mood_force_level>=0 ? (double)cfg.mood_force_level : 0.0);
        }
    }

    //Fluxes_pre + Riemann_Solver over the whole pack. transform_a_to_b_1d is
    //already pack-aware (it loops sd_for_cells_b), so it needs no twin.
    void MHD_Fluxes_pre_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        auto& b0 = blocks[0];
        transform_a_to_b_1d(pv.mhd_U_ader_sp, pv.U_ader_fp_x, b0.sp_to_fp, _x_);
        transform_a_to_b_1d(pv.mhd_U_ader_sp, pv.U_ader_fp_y, b0.sp_to_fp, _y_);
        if(az) transform_a_to_b_1d(pv.mhd_U_ader_sp, pv.U_ader_fp_z, b0.sp_to_fp, _z_);
        if(cfg.rsolver != _rsolver_llf_){
            mhd_face_B_to_fp_b(pv.U_ader_fp_x, pv.Bx_fp_x, _x_);
            mhd_face_B_to_fp_b(pv.U_ader_fp_y, pv.By_fp_y, _y_);
            if(az) mhd_face_B_to_fp_b(pv.U_ader_fp_z, pv.Bz_fp_z, _z_);
        }
        mhd_compute_fluxes_b(pv.U_ader_fp_x, pv.F_ader_fp_x, _x_);
        mhd_compute_fluxes_b(pv.U_ader_fp_y, pv.F_ader_fp_y, _y_);
        if(az) mhd_compute_fluxes_b(pv.U_ader_fp_z, pv.F_ader_fp_z, _z_);
        }
    }

    void MHD_Riemann_Solver_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        mhd_riemann_solver_b(pv.U_ader_fp_x, pv.F_ader_fp_x, _x_);
        mhd_riemann_solver_b(pv.U_ader_fp_y, pv.F_ader_fp_y, _y_);
        if(az) mhd_riemann_solver_b(pv.U_ader_fp_z, pv.F_ader_fp_z, _z_);
        }
    }

    //Compute_E and E_Riemann_Solver over the whole pack, in the same order the
    //per-block MHD_ader methods use.
    void MHD_Compute_E_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        auto& b0 = blocks[0];
        mhd_compute_E_b(pv.Ez_ep_xy, pv.W_sp, pv.Bx_fp_x, pv.By_fp_y, pv.U_sp,
                        b0.sp_to_fp, _z_);
        if(az){
            mhd_compute_E_b(pv.Ey_ep_zx, pv.W_sp, pv.Bz_fp_z, pv.Bx_fp_x, pv.U_sp,
                            b0.sp_to_fp, _y_);
            mhd_compute_E_b(pv.Ex_ep_yz, pv.W_sp, pv.By_fp_y, pv.Bz_fp_z, pv.U_sp,
                            b0.sp_to_fp, _x_);
        }
        }
    }

    void MHD_E_Riemann_batched(){
        if constexpr (!is_mhd) return;
        else {
        const bool az = cfg.active[_z_];
        if(az) mhd_E_riemann_solver_b(pv.Ey_ep_zx, _x_, 4);
        mhd_E_riemann_solver_b(pv.Ez_ep_xy, _x_, 3);
        mhd_E_riemann_solver_b(pv.Ez_ep_xy, _y_, 4);
        if(az){
            mhd_E_riemann_solver_b(pv.Ex_ep_yz, _y_, 3);
            mhd_E_riemann_solver_b(pv.Ex_ep_yz, _z_, 4);
            mhd_E_riemann_solver_b(pv.Ey_ep_zx, _z_, 3);
        }
        }
    }

    void MHD_MOOD_update(){
        if constexpr (!is_mhd) return;
        //3D mixed-level MOOD needs the genuine line-average of the edge EMF along
        //the edge direction, which correct_coarse_fine_fv_emf does not implement.
        //Refuse rather than run a silently non-divergence-free cascade.
        if(forest.max_level()>0 && cfg.active[_z_]){
            if(Master) cout<<"ERROR: MHD MOOD on a mixed-level mesh is 2D-only for now "
                             "(the 3D coarse-fine edge-EMF restriction is not implemented)"<<endl;
            exit(1);
        }
        { PHASE("mood/begin");
          if(mhd_batched(1)) MOOD_begin_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].mood_begin(); }
        //Pack view supplied so this takes the batched transaction-table gather
        //instead of the per-block forest path -- 10.6% of the fenced advance.
        { PHASE("xchg/Exchange_U_fv");
          Exchange_fv_field(&Block::U_old_fv, mhd_batched() ? &pv.U_old_fv : nullptr); }
        { PHASE("mood/after_U_halo");
          if(mhd_batched(2)) MOOD_after_U_halo_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].mood_after_U_halo(); }
        mhd_reduce_nad_gscales();
        for(int rev=0; rev<cfg.max_revs; rev++){
            //Assemble, RECONCILE, then judge. The cascade tests a candidate built
            //from the assembled flux, so the flux has to be single-valued at every
            //block interface before the candidate exists -- otherwise cells demote
            //on an interface mismatch instead of on the solution. This is what the
            //hydro FV path does inside its own loop, and what the Python reference
            //FallbackAMRScheme.mood_loop does with _enforce_flux_consistency().
            { PHASE("mood/assemble");
              if(mhd_batched(4)) MOOD_assemble_batched();
              else for(int b=0;b<nblocks;b++) blocks[b].mood_assemble(); }
            { PHASE("cf/enforce_fv_flux"); enforce_fv_flux_consistency(); }
            { PHASE("cf/enforce_fv_emf");  enforce_fv_emf_consistency(); }
            int demoted = 0;
            { PHASE("mood/detect");
              if(mhd_batched(1024)) demoted = MOOD_detect_batched();
              else for(int b=0;b<nblocks;b++) demoted += blocks[b].mood_detect(); }
            #ifdef MPI
            int g; MPI_Allreduce(&demoted,&g,1,MPI_INT,MPI_SUM,Comm); demoted=g;
            #endif
            if(demoted==0) break;
            Exchange_fv_field_max(&Block::cascade,
                                  mhd_batched() ? &pv.mhd_cascade : nullptr);
        }
        { PHASE("mood/assemble");
          if(mhd_batched(4)) MOOD_assemble_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].mood_assemble(); }
        { PHASE("cf/enforce_fv_flux"); enforce_fv_flux_consistency(); }
        { PHASE("cf/enforce_fv_emf");  enforce_fv_emf_consistency(); }
        { PHASE("mood/commit_assembled");
          if(mhd_batched(8)) MOOD_commit_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].mood_commit_assembled(); }
        //Interior face-B sync is safe only on uniform meshes; mixed-level
        //uses ghost exchange instead (EMF correction owns CF telescoping).
        if(forest.max_level()==0){ PHASE("sync/face_B"); Sync_face_B_mhd(); }
        else { PHASE("xchg/Exchange_face_B"); Exchange_face_B_mhd(); }
    }

    void Advance_hydro(){
        for(int ader=0;ader<n_ader;ader++){
            Solve_fluxes_hydro();
            if(ader<n_ader-1)
                for(int b=0;b<nblocks;b++)
                    blocks[b].Update_prediction(Xd[b].h,Yd[b].h,Zd[b].h);
        }
        Update_solution_hydro();
    }

    void Advance_mhd(){
        if constexpr (!is_mhd) return;
        { PHASE("sd/Fluxes_pre");
          if(mhd_batched(16)) MHD_Fluxes_pre_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].Fluxes_pre(); }
        { PHASE("xchg/Exchange_fp"); Exchange_fp(); }
        //Refresh face-B ghosts and re-project into U so CF fluid Riemann
        //sees B consistent with the staggered field (not the prolonged U-B).
        if(forest.max_level()>0){
            { PHASE("xchg/Exchange_face_B"); Exchange_face_B_mhd(); }
            PHASE("sd/B_to_U");
            if(mhd_batched(64))
                mhd_B_to_U_b(pv.U_sp, pv.Bx_fp_x, pv.By_fp_y, pv.Bz_fp_z,
                             blocks[0].fp_to_sp);
            else
                for(int b=0;b<nblocks;b++)
                    mhd_B_to_U(blocks[b].U_sp, blocks[b].Bx_fp_x, blocks[b].By_fp_y,
                               blocks[b].Bz_fp_z, blocks[b].Tx_, blocks[b].Ty_,
                               blocks[b].Tz_, blocks[b].fp_to_sp);
        }
        { PHASE("sd/Riemann_Solver");
          if(mhd_batched(32)) MHD_Riemann_Solver_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].Riemann_Solver(); }
        if(forest.max_level()>0){
            PHASE("cf/correct_cf_flux");
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) correct_cf_flux_dim(dim);
        }
        { PHASE("sd/Compute_E");
          if(mhd_batched(128)) MHD_Compute_E_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].Compute_E(); }
        { PHASE("xchg/Exchange_E"); Exchange_E_mhd(); }
        { PHASE("sd/E_Riemann_Solver");
          if(mhd_batched(256)) MHD_E_Riemann_batched();
          else for(int b=0;b<nblocks;b++) blocks[b].E_Riemann_Solver(); }
        if(forest.max_level()>0){
            PHASE("cf/correct_cf_emf");
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) correct_coarse_fine_emf(forest, blocks, dim);
            //After BOTH directions, for the same reason as the cascade's version:
            //correct_coarse_fine_emf writes the shared FACE and leaves the patch
            //corner multi-valued in the block diagonal to it.
            if constexpr (is_mhd)
                if(!no_emf_corner())
                    spread_sd_emf_corners_b(pv.Ez_ep_xy, ect_.t, ect_.n);
        }
        if(cfg.fallback) MHD_MOOD_update();
        else {
            PHASE("sd/Update_CT");
            for(int b=0;b<nblocks;b++) blocks[b].Update_CT();
            //Uniform MB: kill round-off face mismatch. Mixed-level: do not
            //overwrite CT interiors (breaks discrete divB); ghost fill via
            //Exchange_face_B_mhd is enough for the next stage's Compute_E.
            if(forest.max_level()==0) Sync_face_B_mhd();
        }
    }

    /////////////////////////////////////////////////////////////////////
    // PhysicsModule interface
    /////////////////////////////////////////////////////////////////////
    void AssembleTasks(Driver* d) override {
        TaskID none(0);
        auto bti = d->tl_map["before_timeintegrator"];
        auto stg = d->tl_map["stagen"];
        auto ati = d->tl_map["after_timeintegrator"];
        auto acy = d->tl_map["after_cycle"];

        if(cfg.integrator==_integrator_rk_ || is_mhd)
            bti->AddTask(&Mesh::TaskSaveState, this, none);

        TaskID pois = stg->AddTask(&Mesh::TaskPoison, this, none);
        TaskID copy = stg->AddTask(&Mesh::TaskCopyCons, this, pois);
        TaskID adv  = stg->AddTask(&Mesh::TaskAdvance,  this, copy);
        TaskID comb = stg->AddTask(&Mesh::TaskCombine,  this, adv);
        if constexpr (is_mhd)
            stg->AddTask(&Mesh::TaskBtoU, this, comb);

        ati->AddTask(&Mesh::TaskConsToPrim, this, none);
        acy->AddTask(&Mesh::TaskAdapt, this, none);
    }


    //SPD_POISON_GHOSTS=1 fills the ghost *elements* of the SD volume arrays
    //with NaN at the top of every stage. SD couples elements only through the
    //flux-point trace on a shared face, so nothing should ever read a volume
    //array outside [NGH, N-NGH): if the answer survives the poison, those
    //ghost elements are dead storage and the blocks can drop them (only the
    //fp face arrays and the FV sub-grid need halos).
    static bool poison_ghosts(){
        static bool v = getenv("SPD_POISON_GHOSTS") != nullptr;
        return v;
    }

    void poison_volume_ghosts(){
        if(!poison_ghosts()) return;
        const double nan_v = std::numeric_limits<double>::quiet_NaN();
        int gx = NGHx, gy = NGHy, gz = NGHz;
        const bool poison_active = getenv("SPD_POISON_ACTIVE") != nullptr;
        auto poison = [&](SD_Solution S){
            if(S.Vector.size()==0) return;
            int nb=S.nb, Nx=S.Nx, Ny=S.Ny, Nz=S.Nz;
            int px=S.nx, py=S.ny, pz=S.nz;
            int nader=S.n_ader, nvar=S.n_var;
            const bool pa = poison_active;
            sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
                KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
                bool ghost = (i<gx || i>=Nx-gx) || (j<gy || j>=Ny-gy)
                          || (k<gz || k>=Nz-gz);
                if(ghost == pa) return;
                const int boff = b*nader;
                for(int t_id=0; t_id<nader; t_id++)
                for(int var=0; var<nvar; var++)
                    S.Vector(boff+t_id,var,k,j,i,kk,jj,ii) = nan_v;
            }, "poison_volume_ghosts");
        };
        poison(pv.U_sp);      poison(pv.W_sp);   poison(pv.W_cv);
        poison(pv.U_cv);      poison(pv.U0_sp);  poison(pv.T_sweep);
        poison(pv.U_ader_sp);
        //Positive control: the fp face arrays are exactly where the exchanged
        //trace lands, so poisoning their ghost slot MUST break the answer.
        //If it does not, the poison itself is not running.
        if(getenv("SPD_POISON_FP")){
            poison(pv.U_ader_fp_x); poison(pv.U_ader_fp_y); poison(pv.U_ader_fp_z);
        }
    }

    TaskStatus TaskPoison(Driver* d, int stage){
        poison_volume_ghosts();
        return TaskStatus::complete;
    }

    void sync_block_dt(){
        for(int b=0;b<nblocks;b++) blocks[b].dt = this->dt;
    }

    //One switch for the RK bookkeeping batching, so the per-block path stays
    //available for a bit-identity A/B: MHD through SPD_MHD_BATCH_MASK bit 512
    //(or SPD_NO_MHD_BATCH), hydro through SPD_NO_RK_BATCH.
    static bool rk_batched(){
        if constexpr (is_mhd) return mhd_batched(512);
        else {
            static const bool v = getenv("SPD_NO_RK_BATCH") == nullptr;
            return v;
        }
    }

    TaskStatus TaskSaveState(Driver* d, int stage){
        //FENCED, and system-agnostic: the state to save comes from
        //Block::rk_state(), so this task does not know or care which system it is
        //running. TaskCombine below walks the SAME list in reverse, which is what
        //keeps a save and its recombination from disagreeing.
        //
        //One copy per ARRAY over the whole pack, not one per block: the blocks'
        //views are slices of it, so this is the same bytes in one launch. The
        //per-block loop was 0.517 s of hydro's 0.771 s fenced total and 1.915 s
        //of MHD's 9.704 s (352 leaves, 109 steps).
        PHASE("rk/save_state");
        if(rk_batched()){
            for(const auto& q : rk_pairs_)
                Kokkos::deep_copy(q.first.Vector, q.second.Vector);
            return TaskStatus::complete;
        }
        for(int b=0;b<nblocks;b++){
            Kokkos::deep_copy(blocks[b].U0_sp.Vector, blocks[b].U_sp.Vector);
            if constexpr (is_mhd){
                Kokkos::deep_copy(blocks[b].B0x_fp_x.Vector, blocks[b].Bx_fp_x.Vector);
                Kokkos::deep_copy(blocks[b].B0y_fp_y.Vector, blocks[b].By_fp_y.Vector);
                Kokkos::deep_copy(blocks[b].B0z_fp_z.Vector, blocks[b].Bz_fp_z.Vector);
            }
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskCopyCons(Driver* d, int stage){
        Region r("TaskCopyCons");
        PHASE("rk/copy_cons");
        sync_block_dt();
        if(rk_batched()){
            //copy_ader is general in n_ader and already spans the pack; for a
            //one-stage (RK) system it is exactly the deep_copy the MHD branch
            //used to do per block.
            Block::copy_ader(pv.U_sp, pv.U_ader_sp);
            //The one real difference between the systems here, named rather than
            //branched on: MHD's compute_E reads W_sp inside the stage.
            if constexpr (Block::prim_at_stage_start)
                Block::primitives_b(pv.U_sp, pv.W_sp);
            return TaskStatus::complete;
        }
        if constexpr (is_hydro){
            blocks[0].copy_ader(pv.U_sp, pv.U_ader_sp);
        } else {
            for(int b=0;b<nblocks;b++){
                Kokkos::deep_copy(blocks[b].U_ader_sp.Vector, blocks[b].U_sp.Vector);
                mhd_compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
            }
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskAdvance(Driver* d, int stage){
        sync_block_dt();
        if constexpr (is_hydro) Advance_hydro();
        else Advance_mhd();
        return TaskStatus::complete;
    }

    TaskStatus TaskCombine(Driver* d, int stage){
        Region r("TaskCombine");
        PHASE("rk/combine");
        if(cfg.integrator==_integrator_rk_ && d->rk_a[stage-1]>0){
            double a = d->rk_a[stage-1];
            if(rk_batched()){
                //The same Block::rk_state() list TaskSaveState walks, with the
                //roles reversed: combine writes the live array from the saved
                //one. combine_solution already takes a pack (nb = U.nb,
                //sd_for_cells_b), which is how the hydro path always spanned
                //every block in one launch while MHD called it per block, four
                //times.
                for(const auto& q : rk_pairs_)
                    combine_solution(q.second, q.first, a);
                return TaskStatus::complete;
            }
            if constexpr (is_hydro){
                combine_solution(pv.U_sp, pv.U0_sp, a);
            } else {
                for(int b=0;b<nblocks;b++){
                    combine_solution(blocks[b].U_sp, blocks[b].U0_sp, a);
                    combine_solution(blocks[b].Bx_fp_x, blocks[b].B0x_fp_x, a);
                    combine_solution(blocks[b].By_fp_y, blocks[b].B0y_fp_y, a);
                    combine_solution(blocks[b].Bz_fp_z, blocks[b].B0z_fp_z, a);
                }
            }
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskBtoU(Driver* d, int stage){
        if constexpr (is_mhd){
            PHASE("rk/B_to_U");
            //mhd_B_to_U_b already existed for the mixed-level path inside
            //Advance_mhd; this call site was still the per-block loop.
            if(rk_batched())
                mhd_B_to_U_b(pv.U_sp, pv.Bx_fp_x, pv.By_fp_y, pv.Bz_fp_z,
                             blocks[0].fp_to_sp);
            else
                for(int b=0;b<nblocks;b++)
                    mhd_B_to_U(blocks[b].U_sp, blocks[b].Bx_fp_x, blocks[b].By_fp_y,
                               blocks[b].Bz_fp_z, blocks[b].Tx_, blocks[b].Ty_,
                               blocks[b].Tz_, blocks[b].fp_to_sp);
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskConsToPrim(Driver* d, int stage){
        Region r("TaskConsToPrim");
        PHASE("rk/cons_to_prim");
        if(rk_batched()){
            //The two systems did the SAME three steps under different names:
            //primitives at the solution points (W_sp feeds the AMR criteria),
            //sp->cv, primitives on the cell averages. One spelling now.
            Block::primitives_b(pv.U_sp, pv.W_sp);
            transform_sp_to_cv_batched(pv.U_sp, pv.U_cv);
            Block::primitives_b(pv.U_cv, pv.W_cv);
            return TaskStatus::complete;
        }
        if constexpr (is_hydro){
            compute_primitives(pv.U_sp, pv.W_sp);
            transform_sp_to_cv_batched(pv.U_sp, pv.U_cv);
            compute_primitives(pv.U_cv, pv.W_cv);
        } else {
            for(int b=0;b<nblocks;b++){
                mhd_compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
                blocks[b].cons_to_prim_cv();
            }
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskAdapt(Driver* d, int stage){
        if(cfg.adapt_interval<=0 || this->n_step%cfg.adapt_interval!=0)
            return TaskStatus::complete;
        PHASE("amr/adapt");
        adapt();
        return TaskStatus::complete;
    }

    double ComputeDt() override {
        Region r("ComputeDt");
        PHASE("rk/compute_dt");
        this->Dt = 1e300;
        bool diverged = false;
        if(rk_batched()){
            //One launch for the whole pack, one name for both systems: the block
            //is an index in the reduction, not a host loop around it. On an AMR
            //forest the loop cost one launch per leaf per step -- and for MHD one
            //MPI_Allreduce per leaf as well, which is why the reduction now lives
            //here instead of inside the kernel wrapper.
            this->Dt = Block::dt_b(pv.W_cv, hx_p, hy_p, hz_p, nu_);
            if(!std::isfinite(this->Dt)) diverged = true;
            //SPD_DT_CHECK=1 cross-checks the packed reduction against the
            //per-block loop it replaced (hydro only: MHD's per-block form
            //reduces internally, so the two are not comparable term by term).
            if constexpr (is_hydro) if(getenv("SPD_DT_CHECK")){
                double ref = 1e300;
                for(int b=0;b<nblocks;b++)
                    ref = std::min(ref, compute_dt(blocks[b].W_cv,
                                    Xd[b].h, Yd[b].h, Zd[b].h, nu_));
                if(Master && std::fabs(ref-this->Dt) > 1e-14*std::fabs(ref))
                    std::cout<<"[dtchk] step "<<this->n_step<<" packed "
                             <<std::setprecision(17)<<this->Dt<<" loop "<<ref
                             <<" nb "<<nblocks<<std::endl;
            }
        } else if constexpr (is_hydro){
            this->Dt = compute_dt_b(pv.W_cv, hx_p, hy_p, hz_p, nu_);
        } else {
            for(int b=0;b<nblocks;b++){
                double db = mhd_compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h);
                if(!std::isfinite(db)) diverged = true;
                this->Dt = std::min(this->Dt, db);
            }
        }
        //ONE reduction for both systems, and for both paths. The per-block MHD
        //loop below still reduces inside each call, so this is a min of minima
        //there; hydro's packed kernel had NO reduction at all, which was a latent
        //MPI bug (dt would not have been global).
        #ifdef MPI
        {
            double g;
            MPI_Allreduce(&this->Dt,&g,1,MPI_DOUBLE,MPI_MIN,Comm);
            this->Dt = g;
        }
        #endif
        if(diverged || !std::isfinite(this->Dt)){
            if(Master)
                std::cout<<std::endl<<"ERROR: non-finite dt at step "<<this->n_step
                    <<" (t = "<<this->t<<"), solution has diverged"<<std::endl;
            Kokkos::finalize();
            exit(1);
        }
        return this->Dt;
    }

    void WriteOutputs() override { Write_outputs(); }

    void prolongate_to_finest(int ib, SD_Solution W, SD_Solution& out,
                              dimension& Xd_b, dimension& Yd_b, dimension& Zd_b,
                              int ox, int oy, int oz, int steps){
        if(steps==0){
            gather_block(W, out, ox, oy, oz);
            return;
        }
        int nchild = 1;
        for(int d=0; d<3; d++) if(cfg.active[d]) nchild *= 2;
        for(int c=0; c<nchild; c++){
            int cx=c&1, cy=(c>>1)&1, cz=(c>>2)&1;
            SD_Solution child("child",1,W.n_var,Zd_b,Yd_b,Xd_b,0,0,0);
            prolongate_block(W, child, amr_P, cx, cy, cz);
            prolongate_to_finest(ib, child, out, Xd_b, Yd_b, Zd_b,
                ox*2+cx*NBx, oy*2+cy*NBy, oz*2+cz*NBz, steps-1);
        }
    }

    //Smallest cell-average of one primitive over every block. The adapt
    //transfer is conservative in rho, rho*v and E, so total_mass() cannot see
    //a bad regrid -- but the pressure those conserved values imply can still
    //come out negative at a shock, and that is what this catches.
    double min_primitive(int var){
        double m = 1e300;
        for(int b=0;b<nblocks;b++){
            SD_Solution W = blocks[b].W_cv;
            int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
            double mb = sd_min_cells(Nz,Ny,Nx,pz,py,px,
                KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& r){
                    double v = W.Vector(0,var,k,j,i,kk,jj,ii);
                    if(v < r) r = v;
                });
            m = std::min(m, mb);
        }
        return m;
    }

    //--------------------------------------------------------------------------
    // Coarse-fine magnetic-flux telescoping: the observable `divb` cannot see.
    //
    // mhd_max_divB computes the divergence PER BLOCK from that block's own face
    // field, and a CT update is locally divergence-free for ANY EMF that is
    // single-valued within the block -- including one that disagrees with the
    // neighbour across a level jump. So `divb` is structurally blind to a
    // coarse-fine EMF mismatch and passes whether or not the correction is right.
    //
    // What is NOT blind is the magnetic flux through the SHARED face, measured
    // from each side. Both CT updates write dB*h = -dt*dE per cell (SD:
    // update_B_solution, FV/cascade: fv_update_B_solution), so over any
    // contiguous run of cells the change in flux telescopes to the EMF at the
    // two ends:
    //     d(flux over the coarse half-face) = -dt*[E_coarse] at the two corners
    //     d(flux over the fine face)        = -dt*[E_fine]   at the same corners
    // The two sides can only stay in step if the EMF is single-valued at those
    // two corners -- exactly what a coarse-fine EMF correction exists to achieve,
    // and what the cascade's FV-lattice EMF had nothing enforcing before step 5.
    //
    // The mismatch VALUE is not the signal: coarse and fine carry different
    // polynomial representations of the same field, so it is nonzero at t=0 and
    // means nothing. The DRIFT is the signal: max over interfaces of |m(t)-m(0)|.
    //
    // Every index into either block is located from the physical coordinates in
    // dimension::fv_faces, never from the index arithmetic of the correction
    // under test. A correction that writes the right value in the wrong place,
    // measured with its own arithmetic, reports agreement -- which is how two
    // earlier attempts at this check passed for the wrong reason.
    //
    // 2D only, matching the correction; reads both blocks of a pair directly,
    // as correct_coarse_fine_fv_emf itself does.
    struct CFLabel {
        int dim, side, ib, jb, lvl_c, lvl_f;
        bool operator!=(const CFLabel& o) const {
            return dim!=o.dim || side!=o.side || ib!=o.ib || jb!=o.jb
                || lvl_c!=o.lvl_c || lvl_f!=o.lvl_f;
        }
    };
    std::vector<double> cf_flux_ref_, sl_flux_ref_;
    //The t=0 reference is keyed on WHICH faces it was taken over, not just how
    //many. A regrid can hand back the same interface count over a different set
    //of faces, and comparing against a reference that silently refers to other
    //faces is a drift number that means nothing.
    std::vector<CFLabel> cf_ref_key_, sl_ref_key_;
    std::vector<CFLabel> cf_label_;
    std::vector<double> cf_scale_;
    int cf_flux_bad_ = 0;

    //Interior extent of a block along `dim`, from the FV face coordinates.
    void cf_block_extent(int ib, int dim, double& lo, double& hi){
        dimension& D = (dim==_x_) ? Xd[ib] : (dim==_y_ ? Yd[ib] : Zd[ib]);
        Vector_h f = setup_mirror(D.fv_faces); setup_pull(D.fv_faces, f);
        GHOST_LOCALS;
        const int g = (dim==_x_) ? sghx : (dim==_y_ ? sghy : sghz);
        lo = f(g);
        hi = f(g + D.N*D.n_sp);
    }

    //Integral of the face-normal B over block ib's own boundary face in
    //direction `dim` (its high end if `high`), restricted to the transverse FV
    //cells whose centre lies in (t0,t1). `width` returns the transverse extent
    //actually integrated, so the caller can check the geometry it asked for is
    //the geometry it got.
    //
    //The face field is stored at solution points transversally; sp_to_cv turns
    //those into the FV sub-cell averages that the CT update itself advances, so
    //sum(cv * cell width) is the exact face integral of the same polynomial in
    //both the SD and the cascade path.
    double cf_face_flux(int ib, int dim, bool high, double t0, double t1,
                        double& width){
        width = 0.0;
        if constexpr (!is_mhd) { (void)ib;(void)dim;(void)high;(void)t0;(void)t1; return 0.0; }
        else {
        GHOST_LOCALS;
        SD_Solution B = (dim==_x_) ? blocks[ib].Bx_fp_x : blocks[ib].By_fp_y;
        const int tdim = (dim==_x_) ? _y_ : _x_;
        dimension& Dt = (tdim==_x_) ? Xd[ib] : Yd[ib];
        const bool xnorm = (dim==_x_);
        //p+2 points along the staggered (normal) direction, p+1 transversally,
        //and p+1 is also the number of FV cells per element.
        const int qn   = xnorm ? B.nx : B.ny;
        const int qt   = xnorm ? B.ny : B.nx;
        const int Ne_n = (xnorm ? B.Nx : B.Ny) - 2*(xnorm ? NGHx : NGHy);
        const int gn   = xnorm ? ghx  : ghy;
        const int gt   = xnorm ? ghy  : ghx;
        const int sgt  = xnorm ? sghy : sghx;
        const int e_n  = high ? gn+Ne_n-1 : gn;   //element carrying the face
        const int p_n  = high ? qn-1      : 0;    //flux point on it
        const int kz   = ghz;
        //Transverse FV cells of this block, selected by coordinate.
        const int J0 = sgt, J1 = sgt + Dt.N*Dt.n_sp;
        Vector_h ft = setup_mirror(Dt.fv_faces); setup_pull(Dt.fv_faces, ft);
        int Jlo = J1, Jhi = J0;
        for(int Jc=J0; Jc<J1; Jc++){
            const double c = 0.5*(ft(Jc)+ft(Jc+1));
            if(c>t0 && c<t1){ if(Jc<Jlo) Jlo=Jc; if(Jc+1>Jhi) Jhi=Jc+1; }
        }
        if(Jhi<=Jlo) return 0.0;
        width = ft(Jhi)-ft(Jlo);
        const int JloL = Jlo, qtL = qt, gtL = gt, sgtL = sgt;
        Vector fv = Dt.fv_faces;
        Matrix S  = blocks[ib].sp_to_cv;
        double sum = 0.0;
        Kokkos::parallel_reduce("cf_face_flux", flat_range(0,flat_total(Jhi-Jlo)),
            KOKKOS_LAMBDA(const unsigned idx, double& acc){
                const int Jc  = JloL + (int)idx;
                const int e_t = gtL + (Jc-sgtL)/qtL;
                const int c   = (Jc-sgtL)%qtL;
                double cv = 0.0;
                for(int m=0; m<qtL; m++){
                    const double b = xnorm ? B.Vector(0,0,kz,e_t,e_n,0,m,p_n)
                                           : B.Vector(0,0,kz,e_n,e_t,0,p_n,m);
                    cv += S(c,m)*b;
                }
                acc += cv*(fv(Jc+1)-fv(Jc));
            }, sum);
        return sum;
        }
    }

    //Per-interface flux mismatch (coarse partial flux minus fine flux), one
    //entry per (dim, side, coarse block, fine sub-neighbour). cf_flux_bad_
    //counts pairs whose geometry did not check out; those are excluded, and a
    //nonzero count means the diagnostic is not measuring what it claims.
    //`same_level`: run the identical measurement over SAME-level block
    //interfaces instead of coarse-fine ones. That is the control that decides
    //whether a nonzero drift is physics or instrument. Sync_face_B_mhd is NOT
    //called on a mixed-level mesh, so a same-level face is held together only
    //by its single-valued Riemann EMF -- exactly the condition the coarse-fine
    //correction is supposed to reproduce across a level jump. If the same-level
    //drift is at round-off while the coarse-fine drift is not, the quadrature
    //and the telescoping premise are sound and the level jump is the defect.
    std::vector<double> cf_flux_mismatch(bool same_level=false){
        std::vector<double> m;
        cf_flux_bad_ = 0;
        cf_label_.clear();
        cf_scale_.clear();
        if constexpr (!is_mhd) return m;
        else {
        if(forest.max_level()==0 || cfg.active[_z_]) return m;
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim] || (dim!=_x_ && dim!=_y_)) continue;
            const int tdim = (dim==_x_) ? _y_ : _x_;
            const double L = forest.domain_lim[dim][1]-forest.domain_lim[dim][0];
            for(int side=0; side<2; side++){
                const FaceGroups& g = forest.face_groups[dim][side];
                const size_t nq = same_level ? g.same_ib.size() : g.fi_ib.size();
                for(size_t q=0; q<nq; q++){
                    const int ib = same_level ? g.same_ib[q] : g.fi_ib[q];
                    //Which end of the coarse block this group's face is on --
                    //taken from the group convention, then VERIFIED below
                    //against the coordinate the two faces must share.
                    const bool c_high = (side==1);
                    double clo, chi; cf_block_extent(ib, dim, clo, chi);
                    const double cn = c_high ? chi : clo;
                    const size_t ns = same_level ? 1 : g.fi_jb[q].size();
                    for(size_t s=0; s<ns; s++){
                        const int jb = same_level ? g.same_jb[q] : g.fi_jb[q][s];
                        double flo, fhi; cf_block_extent(jb, dim, flo, fhi);
                        const double fn = c_high ? flo : fhi;
                        const double d = fabs(cn-fn);
                        //Same face, or the same face across a periodic wrap.
                        if(d > 1e-10*L && fabs(d-L) > 1e-10*L){ cf_flux_bad_++; continue; }
                        double t0, t1; cf_block_extent(jb, tdim, t0, t1);
                        double wc, wf;
                        const double fc = cf_face_flux(ib, dim,  c_high, t0, t1, wc);
                        const double ff = cf_face_flux(jb, dim, !c_high,
                                                       -1e300, 1e300, wf);
                        //The coarse cells selected must cover exactly the fine
                        //block's transverse span, and the fine face must be
                        //that same span. Either failing means the halves are
                        //not the halves this is claiming to compare.
                        const double w = t1-t0;
                        if(fabs(wc-w) > 1e-10*w || fabs(wf-w) > 1e-10*w){
                            cf_flux_bad_++; continue;
                        }
                        m.push_back(fc-ff);
                        cf_label_.push_back({dim, side, ib, jb,
                                             forest.blocks[ib].level,
                                             forest.blocks[jb].level});
                        cf_scale_.push_back(std::max(fabs(fc), fabs(ff)));
                    }
                }
            }
        }
        return m;
        }
    }

    //max |m(t) - m(0)| over coarse-fine interfaces. Negative when there is
    //nothing to measure (uniform mesh, 3D, hydro) or when the interface set
    //changed under us -- a regrid invalidates the t=0 reference, so the drift
    //is only meaningful on a static mesh.
    double mhd_cf_flux_drift(int& n_iface, bool same_level=false){
        n_iface = 0;
        //Two small reductions plus a few coordinate mirrors per interface, once
        //per output. That is negligible next to writing the output itself on
        //the meshes the suite runs, but a production AMR mesh can carry
        //thousands of interfaces, so leave a way to switch it off without a
        //rebuild rather than capping the interface count silently.
        static const bool off = getenv("SPD_NO_CF_FLUX")!=nullptr;
        if(off) return -1.0;
        std::vector<double> m = cf_flux_mismatch(same_level);
        std::vector<double>& ref = same_level ? sl_flux_ref_ : cf_flux_ref_;
        std::vector<CFLabel>& key = same_level ? sl_ref_key_ : cf_ref_key_;
        n_iface = (int)m.size();
        if(m.empty()) return -1.0;
        bool same_faces = (key.size() == m.size());
        for(size_t i=0; same_faces && i<key.size(); i++)
            if(key[i] != cf_label_[i]) same_faces = false;
        if(!same_faces){ ref = m; key = cf_label_; return -1.0; }
        double worst = 0.0;
        static const bool verb = getenv("SPD_CF_VERBOSE")!=nullptr;
        for(size_t i=0; i<m.size(); i++){
            const double d = fabs(m[i]-ref[i]);
            worst = std::max(worst, d);
            if(verb && Master){
                const CFLabel& L = cf_label_[i];
                std::cout<<"    "<<(same_level?"sl":"cf")<<"["<<i<<"]"
                         <<" dim="<<L.dim<<" side="<<L.side
                         <<" b="<<L.ib<<"("<<L.lvl_c<<")"
                         <<"/"<<L.jb<<"("<<L.lvl_f<<")"
                         <<std::scientific<<std::setprecision(3)
                         <<"  |F|="<<cf_scale_[i]<<"  drift="<<d
                         <<"  rel="<<(cf_scale_[i]>0 ? d/cf_scale_[i] : 0.0)
                         <<std::defaultfloat<<std::endl;
            }
        }
        return worst;
    }

    double total_mass(){
        double M=0;
        for(int b=0;b<nblocks;b++){
            if constexpr (is_hydro)
                M += blocks[b].fv_mass(blocks[b].W_cv, Xd[b], Yd[b], Zd[b]);
            else {
                //Density CV average * element volume (MHD has no fv_mass helper).
                //By value: a reference captured into a device lambda leaves a
                //host pointer in the closure, which CUDA rejects outright.
                SD_Solution W = blocks[b].W_cv;
                int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
                int qx=px, qy=py, qz=pz;
                Vector fx = Xd[b].fv_faces, fy = Yd[b].fv_faces, fz = Zd[b].fv_faces;
                bool ay=cfg.active[_y_], az=cfg.active[_z_];
                GHOST_LOCALS;
                double mass = sd_sum_active_cells(Nz,Ny,Nx,pz,py,px,
                    KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& sum){
                        double V = fx(I+1)-fx(I);
                        if(ay) V *= fy(J+1)-fy(J);
                        if(az) V *= fz(K+1)-fz(K);
                        sum += W.Vector(0,0,k,j,i,kk,jj,ii)*V;
                    });
                M += mass;
            }
        }
        return M;
    }

    void Write_outputs(){
        //The phase report used to live inside the is_mhd branch below, so a
        //hydro run could never print one -- which is why comparing the two
        //systems' phase tables silently compared one table against nothing.
        phase_times_.report("cumulative");
        if constexpr (is_mhd){
            double divB = 0.0;
            for(int b=0;b<nblocks;b++)
                divB = std::max(divB,
                    mhd_max_divB(blocks[b].Bx_fp_x, blocks[b].By_fp_y, blocks[b].Bz_fp_z,
                                 blocks[b].dfp_to_sp, Xd[b].h, Yd[b].h, Zd[b].h));
            if(Master)
                std::cout<<std::endl<<"OUTPUT "<<this->n_output
                         <<"  max|divB| = "<<divB<<std::endl;
            //Coarse-fine flux telescoping, which divB above cannot see.
            int n_iface = 0, n_same = 0;
            const double cfd = mhd_cf_flux_drift(n_iface, false);
            const int cf_bad = cf_flux_bad_;
            const double sld = mhd_cf_flux_drift(n_same, true);
            if(Master && n_iface>0){
                std::cout<<"  CF flux drift = "<<cfd<<"  over "<<n_iface
                         <<" interfaces, "<<cf_bad<<" bad"
                         <<"  | same-level control = "<<sld
                         <<" over "<<n_same;
                //The patch-corner spread that the same-level control exists
                //to measure: how many corner points it makes single-valued, and
                //how many of those are the unusual multi-level kind.
                if(getenv("SPD_CF_VERBOSE")) std::cout<<"  | "<<ect_.n<<" emf corners";
                if(emf_corner_odd_>0)
                    std::cout<<"  | "<<emf_corner_odd_<<" multi-level corners";
                //A negative drift is not a measurement: either this is the
                //first output (the t=0 reference is only now being taken) or a
                //regrid changed the interface set and the old reference no
                //longer refers to the same faces.
                if(cfd<0 || sld<0) std::cout<<"   [n/a: reference (re)set]";
                std::cout<<std::endl;
            }
        } else if(Master){
            std::cout<<std::endl<<"OUTPUT "<<this->n_output<<std::endl;
        }
        if(Master){
            std::ofstream f(output_folder()+"mass.txt",
                            this->n_output==0 ? std::ios::trunc : std::ios::app);
            f<<std::setprecision(17)<<this->t<<" "<<total_mass()<<std::endl;
        }
        int M = forest.max_level();
        if(M>0){
            Kokkos::deep_copy(W_glob.Vector, 0.0);
            for(int ib=0; ib<nblocks; ib++){
                int steps = M - forest.blocks[ib].level;
                int ox = forest.blocks[ib].logical[0]*NBx;
                int oy = forest.active[1] ? forest.blocks[ib].logical[1]*NBy : 0;
                int oz = forest.active[2] ? forest.blocks[ib].logical[2]*NBz : 0;
                prolongate_to_finest(ib, blocks[ib].W_cv, W_glob,
                                   Xd[ib], Yd[ib], Zd[ib], ox, oy, oz, steps);
            }
        } else {
            int nbx=forest.N_base[0], nby=forest.N_base[1];
            for(int ib=0; ib<nblocks; ib++){
                int bx=ib%nbx, by=(ib/nbx)%nby, bz=ib/(nbx*nby);
                gather_block(blocks[ib].W_cv,W_glob,bx*NBx,by*NBy,bz*NBz);
            }
        }
        Kokkos::fence();
        Write(W_glob,this->n_output);
        Write_amr_blocks(forest, this->n_output, NBx, NBy, NBz);
        this->n_output++;
    }

    std::map<BlockForest::BlockKey,int> snapshot_keys(){
        std::map<BlockForest::BlockKey,int> m;
        for(int ib=0; ib<nblocks; ib++) m[forest.block_key(ib)] = ib;
        return m;
    }

    //Per-block snapshot for adapt transfer. Hydro uses U only; MHD also packs
    //face-staggered B so refine/derefine preserve the CT field (not re-inited
    //from the vector potential in make_block).
    struct BlockSnap {
        SD_Solution U;
        SD_Solution Bx, By, Bz;
    };

    BlockSnap make_empty_snap(int ib, const char* tag){
        BlockSnap s;
        s.U.init(std::string(tag)+"_U", blocks[ib].n_ader, blocks[ib].nvar,
                 Zd[ib], Yd[ib], Xd[ib], 0, 0, 0);
        if constexpr (is_mhd){
            s.Bx.init(std::string(tag)+"_Bx", 1, 1, Zd[ib], Yd[ib], Xd[ib], 0, 0, 1);
            s.By.init(std::string(tag)+"_By", 1, 1, Zd[ib], Yd[ib], Xd[ib], 0, 1, 0);
            s.Bz.init(std::string(tag)+"_Bz", 1, 1, Zd[ib], Yd[ib], Xd[ib], 1, 0, 0);
        }
        return s;
    }

    void capture_block_snap(int ib, BlockSnap& s){
        Kokkos::deep_copy(s.U.Vector, blocks[ib].U_sp.Vector);
        if constexpr (is_mhd){
            Kokkos::deep_copy(s.Bx.Vector, blocks[ib].Bx_fp_x.Vector);
            Kokkos::deep_copy(s.By.Vector, blocks[ib].By_fp_y.Vector);
            Kokkos::deep_copy(s.Bz.Vector, blocks[ib].Bz_fp_z.Vector);
        }
    }

    //p = 0 carries no sub-element polynomial for amr_P to interpolate, so the
    //matrix path is piecewise-constant injection there; take the limited-linear
    //reconstruction instead (see prolongate_block_lim). p >= 1 keeps the
    //Lagrange operator, which is exact on degree-p data.
    void prolongate_snap(const BlockSnap& src, BlockSnap& dst,
                         int cx, int cy, int cz, int ib_mat){
        if(Xd[0].p == 0) prolongate_block_lim(src.U, dst.U, cx, cy, cz);
        else             prolongate_block(src.U, dst.U, amr_P, cx, cy, cz);
        if constexpr (is_mhd){
            prolongate_block_face_B(src.Bx, src.By, src.Bz,
                                    dst.Bx, dst.By, dst.Bz,
                                    amr_P, blocks[ib_mat].sp_to_cv,
                                    blocks[ib_mat].cv_to_sp, cx, cy, cz);
        }
    }

    void restrict_snap_child(const BlockSnap& fine, BlockSnap& coarse,
                             int cx, int cy, int cz, int ib_mat){
        restrict_block(fine.U, coarse.U, amr_RF, cx, cy, cz);
        if constexpr (is_mhd){
            restrict_block_face_B_2d(fine.Bx, fine.By, fine.Bz,
                                     coarse.Bx, coarse.By, coarse.Bz,
                                     amr_RF, blocks[ib_mat].sp_to_cv,
                                     blocks[ib_mat].cv_to_sp, cx, cy, cz);
        }
    }

    void install_snap(int ib, const BlockSnap& s){
        Kokkos::deep_copy(blocks[ib].U_sp.Vector, s.U.Vector);
        if constexpr (is_mhd){
            Kokkos::deep_copy(blocks[ib].Bx_fp_x.Vector, s.Bx.Vector);
            Kokkos::deep_copy(blocks[ib].By_fp_y.Vector, s.By.Vector);
            Kokkos::deep_copy(blocks[ib].Bz_fp_z.Vector, s.Bz.Vector);
        }
    }

    //After face B is in place (copied / prolongated / restricted), project onto
    //cell-centered primitives and rebuild conservatives so total energy uses
    //the transferred magnetic field. Face B is not re-inited from A.
    void finish_block_ic(int ib){
        if constexpr (is_hydro){
            compute_primitives(blocks[ib].U_sp, blocks[ib].W_sp);
            blocks[ib].cons_to_prim_cv();
        } else {
            mhd_compute_primitives(blocks[ib].U_sp, blocks[ib].W_sp);
            mhd_B_to_U(blocks[ib].W_sp, blocks[ib].Bx_fp_x, blocks[ib].By_fp_y,
                       blocks[ib].Bz_fp_z, blocks[ib].Tx_, blocks[ib].Ty_,
                       blocks[ib].Tz_, blocks[ib].fp_to_sp);
            mhd_compute_conservatives(blocks[ib].W_sp, blocks[ib].U_sp);
            blocks[ib].cons_to_prim_cv();
        }
    }

    //Elements the prolongation limiter had to collapse, last transfer, and
    //those whose own mean was still inadmissible (which this unit cannot fix).
    int prolong_limited = 0;
    int prolong_unfixable = 0;

    void transfer_from_snapshot(std::map<BlockForest::BlockKey,int>& key_to_ib,
                                std::vector<BlockSnap>& snap){
        prolong_limited = 0;
        prolong_unfixable = 0;
        auto find_ancestor = [&](BlockForest::BlockKey key,
                                 std::vector<std::array<int,3>>& chain)->int{
            BlockForest::BlockKey k = key;
            while(std::get<0>(k) > 0){
                chain.push_back({(int)(std::get<1>(k)%2),
                                 (int)(std::get<2>(k)%2),
                                 (int)(std::get<3>(k)%2)});
                k = {std::get<0>(k)-1, std::get<1>(k)/2, std::get<2>(k)/2, std::get<3>(k)/2};
                auto it = key_to_ib.find(k);
                if(it != key_to_ib.end()) return it->second;
            }
            return -1;
        };

        for(int ib=0; ib<nblocks; ib++){
            BlockForest::BlockKey key = forest.block_key(ib);
            auto it = key_to_ib.find(key);
            if(it != key_to_ib.end()){
                install_snap(ib, snap[it->second]);
                continue;
            }
            std::vector<std::array<int,3>> chain;
            int pib = find_ancestor(key, chain);
            if(pib >= 0){
                BlockSnap cur = snap[pib];
                for(int s=(int)chain.size()-1; s>=1; s--){
                    BlockSnap tmp = make_empty_snap(ib, "pro");
                    prolongate_snap(cur, tmp, chain[s][0], chain[s][1], chain[s][2], ib);
                    cur = tmp;
                }
                BlockSnap dst = make_empty_snap(ib, "pro_dst");
                prolongate_snap(cur, dst, chain[0][0], chain[0][1], chain[0][2], ib);
                install_snap(ib, dst);
                //The interpolation is unlimited, so at a discontinuity it can
                //hand the new fine block a state no update can recover from.
                if constexpr (is_hydro){
                    blocks[ib].transform_sp_to_cv(blocks[ib].U_sp, blocks[ib].U_cv);
                    //The DMP pass is a measured dead end, off by default:
                    //at p >= 1 the Lagrange prolongation is a LOSSLESS change
                    //of representation (it re-samples the coarse element's own
                    //polynomial), so it injects nothing for a limiter to find
                    //and the pass only deletes real structure. See
                    //limit_prolongation_dmp.
                    int nlim = 0;
                    if(cfg.amr_prolong_dmp && Xd[0].p > 0)
                        nlim += limit_prolongation_dmp(blocks[ib].U_cv,
                            Xd[ib].fv_faces, Yd[ib].fv_faces, Zd[ib].fv_faces);
                    nlim += limit_prolongation(blocks[ib].U_cv,
                        Xd[ib].fv_faces, Yd[ib].fv_faces, Zd[ib].fv_faces,
                        cfg.gamma, &prolong_unfixable);
                    prolong_limited += nlim;
                    if(nlim)
                        blocks[ib].transform_cv_to_sp(blocks[ib].U_cv,
                                                      blocks[ib].U_sp);
                }
                continue;
            }
            int n_sib = 1;
            for(int d=0; d<3; d++) if(forest.active[d]) n_sib *= 2;
            std::vector<int> sibs;
            for(int s=0; s<n_sib; s++){
                int cx=s&1, cy=(s>>1)&1, cz=(s>>2)&1;
                BlockForest::BlockKey ckey = {
                    std::get<0>(key)+1,
                    2*std::get<1>(key)+cx,
                    2*std::get<2>(key)+cy,
                    2*std::get<3>(key)+cz};
                auto cit = key_to_ib.find(ckey);
                if(cit != key_to_ib.end()) sibs.push_back(cit->second);
            }
            if((int)sibs.size()==n_sib){
                BlockSnap acc = make_empty_snap(ib, "acc");
                for(int s=0; s<n_sib; s++){
                    int cx=s&1, cy=(s>>1)&1, cz=(s>>2)&1;
                    restrict_snap_child(snap[sibs[s]], acc, cx, cy, cz, ib);
                }
                install_snap(ib, acc);
                continue;
            }
            if(Master)
                std::cout<<std::endl<<"ERROR: adapt could not transfer block "<<ib
                    <<" (level "<<std::get<0>(key)<<", logical "<<std::get<1>(key)<<","
                    <<std::get<2>(key)<<","<<std::get<3>(key)<<"): no snapshot ancestor "
                    <<"and only "<<sibs.size()<<" of "<<n_sib<<" children"<<std::endl;
            Kokkos::finalize();
            exit(1);
        }
    }

    void report_divb(const char* where){
        if constexpr (is_mhd){
            if(!getenv("SPD_DIVB_DEBUG")) return;
            double d = 0.0;
            int worst = -1;
            for(int b=0;b<nblocks;b++){
                double v = mhd_max_divB(blocks[b].Bx_fp_x, blocks[b].By_fp_y,
                                        blocks[b].Bz_fp_z, blocks[b].dfp_to_sp,
                                        Xd[b].h, Yd[b].h, Zd[b].h);
                if(v > d){ d = v; worst = b; }
            }
            if(Master)
                std::cout<<"\n[divb] "<<where<<": "<<d<<" (block "<<worst
                         <<" level "<<(worst<0?-1:forest.blocks[worst].level)
                         <<", nblocks "<<nblocks<<")"<<std::endl;
        }
    }

    //Drive the initial refinement to convergence BEFORE the first step, the way
    //Athena++ does (Mesh::Initialize: `do { ProblemGenerator; CheckRefinementCondition;
    //AMR } while (nbtotal changed)`, mesh.cpp ~1427-1722).
    //
    //Why this matters, measured on fig-21 KH at 1024^2 DoF: adapt() gains ONE level
    //per call (as do Athena++ and AthenaK), so a 3-level run reached its finest level
    //only around step 100-150 of 13,300 and spent that transient under-resolved. KH
    //amplifies the resulting seed exponentially, and no amount of adapt_interval or
    //root resolution recovers it -- both knobs saturate (0.157 and 0.094 rms) while
    //static refinement covering the same region from t=0 gives 0.0034, i.e. 27x
    //better. The transient, not the transfer operators, is the whole gap.
    //
    //New blocks are given ICs RE-EVALUATED at their own resolution
    //(build_block_solvers(run_ic=true)), NOT prolongated coarse data -- Athena++
    //re-runs ProblemGenerator inside its loop for the same reason. Prolongating
    //would hand the fine mesh a degree-p fit of the coarse fit, which is exactly the
    //seed this is meant to remove.
    //
    //Refine only. A criterion asking to DEREFINE the initial conditions is a
    //criterion bug, not a mesh outcome (Athena++ warns on the same condition), so
    //say so rather than acting on it.
    void initial_refine(){
        if(!cfg.amr_initial_refine || cfg.amr_max_level <= 0) return;
        //One pass per level is sufficient (each pass gains at most one level); the
        //+1 lets the loop observe convergence, and it is a hard cap either way.
        for(int pass=0; pass < cfg.amr_max_level + 1; pass++){
            std::vector<int> to_refine;
            std::vector<std::vector<int>> to_derefine;
            tag_blocks(forest, blocks, to_refine, to_derefine,
                       cfg.amr_max_level, cfg.amr_criterion);
            if(pass==0 && !to_derefine.empty() && Master)
                std::cout<<"WARNING: the refinement criterion wants to DEREFINE "
                         <<to_derefine.size()<<" group(s) of the initial conditions;"
                         <<" check the derefine threshold"<<std::endl;
            if(to_refine.empty()) break;
            const int nb_before = forest.Nblocks();
            forest.refine_blocks(to_refine);
            forest.enforce_2to1_balance();
            if(forest.Nblocks() == nb_before) break;   //converged
            //Rebuild with the ICs re-evaluated on the new mesh.
            build_block_solvers(true);
            //Each block re-inits its own face B from the vector potential, so the
            //two sides of a new coarse-fine face disagree until they are
            //reconciled -- adapt() does this after every transfer and the first
            //adapt() used to do it for the initial mesh too. Skipping it left the
            //CT field inconsistent across the new interfaces and collapsed dt a
            //few steps in (mhd_orszag_tang_amr_2d, dt -> 4e-12 of its initial
            //value at step 3).
            if constexpr (is_mhd){
                if(forest.max_level()==0) Sync_face_B_mhd();
                else                      Exchange_face_B_mhd();
                for(int ib=0; ib<nblocks; ib++) finish_block_ic(ib);
            }
            if(Master)
                std::cout<<"initial refine pass "<<pass+1<<": "<<nb_before<<" -> "
                         <<nblocks<<" blocks, max_level = "<<forest.max_level()<<std::endl;
        }
        recompute_dt();
    }

    void adapt(){
        if constexpr (is_mhd){
            if(cfg.active[_z_]){
                if(Master)
                    std::cout<<std::endl<<"ERROR: dynamic AMR for 3D MHD is not implemented yet"
                               <<" (face-B prolongate is 2D-only)"<<std::endl;
                exit(1);
            }
        }
        //SPD_ADAPT_MASS=1 brackets every regrid with the conserved mass, so a
        //transfer that loses mass is separated from an evolution that does.
        const bool mass_dbg = getenv("SPD_ADAPT_MASS") != nullptr;
        //Tag first: it only reads the blocks, and a regrid that turns out to be
        //a no-op should not have paid for a snapshot of every block.
        std::vector<int> to_refine;
        std::vector<std::vector<int>> to_derefine;
        tag_blocks(forest, blocks, to_refine, to_derefine,
                   cfg.amr_max_level, cfg.amr_criterion);
        //SPD_NO_DEREFINE=1 keeps every refinement once it is made, which
        //separates a bad derefine (restriction) from a bad refine
        //(prolongation) when a regrid-driven run goes unstable.
        if(getenv("SPD_NO_DEREFINE")) to_derefine.clear();
        const size_t n_ref = to_refine.size(), n_deref_tagged = to_derefine.size();
        //Drop the groups enforce_2to1_balance() would refine straight back
        //before anything is spent on them. The criterion keeps proposing them
        //every adapt -- it scores the solution, it does not know the balance
        //rule -- so without this the regrid tears down 24 blocks, rebuilds the
        //neighbour tables, refines them again and transfers the whole forest,
        //to arrive exactly where it started.
        {
            std::vector<std::vector<int>> keep;
            for(auto &g : to_derefine)
                if(forest.derefine_allowed(g)) keep.push_back(std::move(g));
            to_derefine.swap(keep);
        }
        const size_t n_deref = to_derefine.size();
        if(to_refine.empty() && to_derefine.empty()) return;

        double m_before = mass_dbg ? total_mass() : 0.0;
        double p_before = 0.0, d_before = 0.0;
        if(mass_dbg && is_hydro){
            p_before = min_primitive(_p_);
            d_before = min_primitive(_d_);
        }
        //At p = 0 the transfer reconstructs a limited slope from each coarse
        //block's neighbours, so the ghosts the snapshots carry have to describe
        //this state and not the last stage's. Free for p >= 1, whose matrix
        //prolongation reads the element only.
        if(Xd[0].p == 0)
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) Exchange_sd_field(&Block::U_sp, dim);
        std::vector<BlockSnap> snap(nblocks);
        for(int ib=0; ib<nblocks; ib++){
            snap[ib] = make_empty_snap(ib, "snap");
            capture_block_snap(ib, snap[ib]);
        }
        auto key_to_ib = snapshot_keys();

        int old_M = forest.max_level();
        auto deref_keys = forest.keys_of(to_derefine);
        if(!to_refine.empty()) forest.refine_blocks(to_refine);
        const int nb_ref = forest.Nblocks();
        if(!deref_keys.empty()) forest.derefine_blocks_keys(deref_keys);
        const int nb_deref = forest.Nblocks();
        forest.enforce_2to1_balance();
        const int nb_bal = forest.Nblocks();

        //The forest the next step runs on must have every level jump in a
        //group; a dropped face silently skips both its ghost fill and its
        //flux correction.
        if(forest.dropped_faces && Master)
            std::cout<<std::endl<<"WARNING: step "<<this->n_step<<": "
                     <<forest.dropped_faces<<" coarse-fine face(s) left ungrouped"
                     <<" after enforce_2to1_balance"<<std::endl;

        build_block_solvers(false);
        transfer_from_snapshot(key_to_ib, snap);
        if constexpr (is_mhd) report_divb("after transfer");
        if constexpr (is_mhd){
            if(forest.max_level()==0) Sync_face_B_mhd();
            else Exchange_face_B_mhd();
        }
        if constexpr (is_mhd) report_divb("after exchange");
        for(int ib=0; ib<nblocks; ib++) finish_block_ic(ib);
        if(mass_dbg && Master){
            double m_after = total_mass();
            //nblocks through each stage: a derefine that balance immediately
            //puts back shows up here as deref-> dropping and balance-> restoring.
            std::cout<<std::endl<<"[adapt] step "<<this->n_step
                     <<" nblocks "<<nblocks
                     <<" (ref "<<n_ref<<"->"<<nb_ref
                     <<", deref "<<n_deref<<"/"<<n_deref_tagged<<"->"<<nb_deref
                     <<", bal->"<<nb_bal<<")"
                     <<" mass "<<std::setprecision(17)<<m_before<<" -> "<<m_after
                     <<"  rel "<<std::setprecision(6)
                     <<(m_before!=0.0 ? (m_after-m_before)/m_before : 0.0);
            if(is_hydro)
                std::cout<<"  minP "<<p_before<<" -> "<<min_primitive(_p_)
                         <<"  minRho "<<d_before<<" -> "<<min_primitive(_d_)
                         <<"  prolong_limited "<<prolong_limited
                         <<(prolong_unfixable ? "  UNFIXABLE " : "")
                         <<(prolong_unfixable ? std::to_string(prolong_unfixable) : "");
            std::cout<<std::endl;
        }
        recompute_dt();
        if(forest.max_level() != old_M)
            init_W_glob(Xg, Yg, Zg, x_fp_);
    }
};

#endif  // MESH_HPP_
