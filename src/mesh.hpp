#ifndef MESH_HPP_
#define MESH_HPP_

#include <array>
#include <map>
#include <type_traits>
#include <vector>

#include "forest.hpp"
#include "amr_criteria.hpp"

//Block-forest mesh driver: uniform multiblock and mixed-level AMR (hydro +
//true-2D MHD). Derives from PhysicsModule and registers mesh-orchestrated
//tasks into Driver.
template<typename Block>
struct Mesh : public PhysicsModule {
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

    Block make_block(const dimension& Xdim, const dimension& Ydim, const dimension& Zdim){
        if constexpr (is_hydro)
            return Hydro_ader(comm_,p_,Xdim,Ydim,Zdim,x_,w_,x_sp_,x_fp_,nu_,beta_,false);
        else
            return MHD_ader(comm_,p_,Xdim,Ydim,Zdim,x_,w_,x_sp_,x_fp_,false);
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

    void build_block_solvers(){
        blocks.clear(); Xd.clear(); Yd.clear(); Zd.clear();
        nblocks = forest.Nblocks();
        for(int ib=0; ib<nblocks; ib++){
            const MeshBlock& b = forest.blocks[ib];
            Xd.emplace_back(x_dim_for_block(b, p_, x_fp_));
            Yd.emplace_back(y_dim_for_block(b, p_, x_fp_));
            Zd.emplace_back(z_dim_for_block(b, p_, x_fp_));
            blocks.push_back(make_block(Xd[ib], Yd[ib], Zd[ib]));
        }
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
                blocks[b].transform_sp_to_cv(blocks[b].W_sp, blocks[b].W_cv);
                this->Dt = std::min(this->Dt,
                    compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h, nu_));
            } else {
                mhd_compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
                blocks[b].transform_sp_to_cv(blocks[b].W_sp, blocks[b].W_cv);
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

    void Exchange_fp(){
        if(nblocks<=1 && forest.max_level()==0) return;
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            if(forest.max_level()==0){
                for(int b=0; b<nblocks; b++){
                    int L,R,tL,tR;
                    neighbors_uniform(b,dim,L,R,tL,tR);
                    block_boundary_sd(fp(b,dim),fp(L,dim),fp(R,dim),tL,tR,dim);
                }
            } else {
                forest_exchange_fp(forest, blocks, dim);
            }
        }
    }

    void Exchange_fv_field(FV_Solution Block::*member){
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            if(forest.max_level()>0){
                forest_exchange_fv(forest, blocks, member, dim);
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

    void Exchange_sd_field(SD_Solution Block::*member, int dim,
                           bool cf_prolong=true){
        if(!cfg.active[dim]) return;
        if(nblocks<=1 && forest.max_level()==0) return;
        if(forest.max_level()>0){
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

    void Exchange_fv_field_max(FV_Solution Block::*member){
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

    void Solve_fluxes_hydro(){
        for(int b=0;b<nblocks;b++) blocks[b].Fluxes_pre();
        Exchange_fp();
        for(int b=0;b<nblocks;b++) blocks[b].Riemann_Solver();
        if(forest.max_level()>0){
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) correct_coarse_fine_flux(forest, blocks, dim);
        }
        #ifdef VISCOSITY
        for(int b=0;b<nblocks;b++) blocks[b].Viscosity(Xd[b].h,Yd[b].h,Zd[b].h);
        Exchange_fp();
        for(int b=0;b<nblocks;b++) blocks[b].Rusanov_Solver();
        #endif
    }

    void FV_Update_solution_hydro(){
        for(int b=0;b<nblocks;b++) blocks[b].FV_begin();
        for(int ader=0;ader<n_ader;ader++){
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_flux_update(ader,Xd[b],Yd[b],Zd[b]);
            Exchange_fv_field(&Block::U_old);
            Exchange_fv_field(&Block::U_new);
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_detect(Xd[b],Yd[b],Zd[b]);
            if(!cfg.muscl_only) Exchange_fv_field(&Block::troubles);
            for(int b=0;b<nblocks;b++) blocks[b].FV_theta();
            if(!cfg.muscl_only) Exchange_fv_field(&Block::theta);
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_blend(ader,Xd[b],Yd[b],Zd[b]);
            if(forest.max_level()>0)
                for(int dim=0; dim<3; dim++)
                    if(cfg.active[dim])
                        correct_coarse_fine_fv_flux(forest, blocks, dim);
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_commit(ader,Xd[b],Yd[b],Zd[b]);
        }
        for(int b=0;b<nblocks;b++) blocks[b].FV_end();
    }

    void Update_solution_hydro(){
        if(cfg.fallback) FV_Update_solution_hydro();
        else for(int b=0;b<nblocks;b++)
            blocks[b].Update_solution(Xd[b].h,Yd[b].h,Zd[b].h);
    }

    void Exchange_E_mhd(){
        if constexpr (!is_mhd) return;
        bool az = cfg.active[_z_];
        //Edge EMF: CF ghosts mirrored (fp P/R unstable); same-level copied.
        Exchange_sd_field(&Block::Ez_ep_xy, _x_, false);
        Exchange_sd_field(&Block::Ez_ep_xy, _y_, false);
        if(az){
            Exchange_sd_field(&Block::Ey_ep_zx, _x_, false);
            Exchange_sd_field(&Block::Ey_ep_zx, _z_, false);
            Exchange_sd_field(&Block::Ex_ep_yz, _y_, false);
            Exchange_sd_field(&Block::Ex_ep_yz, _z_, false);
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
        Exchange_sd_field(&Block::Bx_fp_x, _x_, true);
        Exchange_sd_field(&Block::By_fp_y, _y_, true);
        if(cfg.active[_z_])
            Exchange_sd_field(&Block::Bz_fp_z, _z_, true);
    }

    void MHD_MOOD_update(){
        if constexpr (!is_mhd) return;
        for(int b=0;b<nblocks;b++) blocks[b].mood_begin();
        Exchange_fv_field(&Block::U_old_fv);
        for(int b=0;b<nblocks;b++) blocks[b].mood_after_U_halo();
        for(int rev=0; rev<cfg.max_revs; rev++){
            int demoted = 0;
            for(int b=0;b<nblocks;b++) demoted += blocks[b].mood_revision();
            #ifdef MPI
            int g; MPI_Allreduce(&demoted,&g,1,MPI_INT,MPI_SUM,Comm); demoted=g;
            #endif
            if(demoted==0) break;
            Exchange_fv_field_max(&Block::cascade);
        }
        for(int b=0;b<nblocks;b++) blocks[b].mood_commit();
        //Interior face-B sync is safe only on uniform meshes; mixed-level
        //uses ghost exchange instead (EMF correction owns CF telescoping).
        if(forest.max_level()==0) Sync_face_B_mhd();
        else Exchange_face_B_mhd();
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
        for(int b=0;b<nblocks;b++) blocks[b].Fluxes_pre();
        Exchange_fp();
        //Refresh face-B ghosts and re-project into U so CF fluid Riemann
        //sees B consistent with the staggered field (not the prolonged U-B).
        if(forest.max_level()>0){
            Exchange_face_B_mhd();
            for(int b=0;b<nblocks;b++)
                mhd_B_to_U(blocks[b].U_sp, blocks[b].Bx_fp_x, blocks[b].By_fp_y,
                           blocks[b].Bz_fp_z, blocks[b].Tx_, blocks[b].Ty_, blocks[b].Tz_,
                           blocks[b].fp_to_sp);
        }
        for(int b=0;b<nblocks;b++) blocks[b].Riemann_Solver();
        if(forest.max_level()>0){
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) correct_coarse_fine_flux(forest, blocks, dim);
        }
        for(int b=0;b<nblocks;b++) blocks[b].Compute_E();
        Exchange_E_mhd();
        for(int b=0;b<nblocks;b++) blocks[b].E_Riemann_Solver();
        if(forest.max_level()>0){
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim]) correct_coarse_fine_emf(forest, blocks, dim);
        }
        if(cfg.fallback) MHD_MOOD_update();
        else {
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

        TaskID copy = stg->AddTask(&Mesh::TaskCopyCons, this, none);
        TaskID adv  = stg->AddTask(&Mesh::TaskAdvance,  this, copy);
        TaskID comb = stg->AddTask(&Mesh::TaskCombine,  this, adv);
        if constexpr (is_mhd)
            stg->AddTask(&Mesh::TaskBtoU, this, comb);

        ati->AddTask(&Mesh::TaskConsToPrim, this, none);
        acy->AddTask(&Mesh::TaskAdapt, this, none);
    }

    void sync_block_dt(){
        for(int b=0;b<nblocks;b++) blocks[b].dt = this->dt;
    }

    TaskStatus TaskSaveState(Driver* d, int stage){
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
        sync_block_dt();
        for(int b=0;b<nblocks;b++){
            if constexpr (is_hydro)
                blocks[b].copy_ader(blocks[b].U_sp, blocks[b].U_ader_sp);
            else {
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
        if(cfg.integrator==_integrator_rk_ && d->rk_a[stage-1]>0){
            for(int b=0;b<nblocks;b++){
                combine_solution(blocks[b].U_sp, blocks[b].U0_sp, d->rk_a[stage-1]);
                if constexpr (is_mhd){
                    combine_solution(blocks[b].Bx_fp_x, blocks[b].B0x_fp_x, d->rk_a[stage-1]);
                    combine_solution(blocks[b].By_fp_y, blocks[b].B0y_fp_y, d->rk_a[stage-1]);
                    combine_solution(blocks[b].Bz_fp_z, blocks[b].B0z_fp_z, d->rk_a[stage-1]);
                }
            }
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskBtoU(Driver* d, int stage){
        if constexpr (is_mhd){
            for(int b=0;b<nblocks;b++)
                mhd_B_to_U(blocks[b].U_sp, blocks[b].Bx_fp_x, blocks[b].By_fp_y,
                           blocks[b].Bz_fp_z, blocks[b].Tx_, blocks[b].Ty_,
                           blocks[b].Tz_, blocks[b].fp_to_sp);
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskConsToPrim(Driver* d, int stage){
        for(int b=0;b<nblocks;b++){
            if constexpr (is_hydro){
                compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
                blocks[b].transform_sp_to_cv(blocks[b].W_sp, blocks[b].W_cv);
            } else {
                mhd_compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
                blocks[b].transform_sp_to_cv(blocks[b].W_sp, blocks[b].W_cv);
            }
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskAdapt(Driver* d, int stage){
        if(cfg.adapt_interval<=0 || this->n_step%cfg.adapt_interval!=0)
            return TaskStatus::complete;
        adapt();
        return TaskStatus::complete;
    }

    double ComputeDt() override {
        this->Dt = 1e300;
        bool diverged = false;
        for(int b=0;b<nblocks;b++){
            double db;
            if constexpr (is_hydro)
                db = compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h, nu_);
            else
                db = mhd_compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h);
            if(!std::isfinite(db)) diverged = true;
            this->Dt = std::min(this->Dt, db);
        }
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

    double total_mass(){
        double M=0;
        for(int b=0;b<nblocks;b++){
            if constexpr (is_hydro)
                M += blocks[b].fv_mass(blocks[b].W_cv, Xd[b], Yd[b], Zd[b]);
            else {
                //Density CV average * element volume (MHD has no fv_mass helper)
                SD_Solution& W = blocks[b].W_cv;
                int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
                int qx=px, qy=py, qz=pz;
                Vector fx = Xd[b].fv_faces, fy = Yd[b].fv_faces, fz = Zd[b].fv_faces;
                bool ay=cfg.active[_y_], az=cfg.active[_z_];
                GHOST_LOCALS;
                double mass=0;
                Kokkos::parallel_reduce("mesh_mhd_mass",
                    Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
                        {NGHz,NGHy,NGHx,0,0,0},{Nz-NGHz,Ny-NGHy,Nx-NGHx,pz,py,px}),
                    KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& sum){
                        double V = fx(I+1)-fx(I);
                        if(ay) V *= fy(J+1)-fy(J);
                        if(az) V *= fz(K+1)-fz(K);
                        sum += W.Vector(0,0,k,j,i,kk,jj,ii)*V;
                    }, mass);
                M += mass;
            }
        }
        return M;
    }

    void Write_outputs(){
        if constexpr (is_mhd){
            double divB = 0.0;
            for(int b=0;b<nblocks;b++)
                divB = std::max(divB,
                    mhd_max_divB(blocks[b].Bx_fp_x, blocks[b].By_fp_y, blocks[b].Bz_fp_z,
                                 blocks[b].dfp_to_sp, Xd[b].h, Yd[b].h, Zd[b].h));
            if(Master)
                std::cout<<std::endl<<"OUTPUT "<<this->n_output
                         <<"  max|divB| = "<<divB<<std::endl;
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

    void prolongate_snap(const BlockSnap& src, BlockSnap& dst,
                         int cx, int cy, int cz, int ib_mat){
        prolongate_block(src.U, dst.U, amr_P, cx, cy, cz);
        if constexpr (is_mhd){
            prolongate_block_face_B(src.Bx, src.By, src.Bz,
                                    dst.Bx, dst.By, dst.Bz,
                                    amr_P, blocks[ib_mat].sp_to_cv,
                                    blocks[ib_mat].cv_to_sp, cx, cy, cz);
        }
    }

    void restrict_snap_child(const BlockSnap& fine, BlockSnap& coarse,
                             int cx, int cy, int cz){
        restrict_block(fine.U, coarse.U, amr_RF, cx, cy, cz);
        if constexpr (is_mhd){
            restrict_block_face_B(fine.Bx, coarse.Bx, amr_RF, _x_, cx, cy, cz);
            restrict_block_face_B(fine.By, coarse.By, amr_RF, _y_, cx, cy, cz);
            restrict_block_face_B(fine.Bz, coarse.Bz, amr_RF, _z_, cx, cy, cz);
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
            blocks[ib].transform_sp_to_cv(blocks[ib].W_sp, blocks[ib].W_cv);
        } else {
            mhd_compute_primitives(blocks[ib].U_sp, blocks[ib].W_sp);
            mhd_B_to_U(blocks[ib].W_sp, blocks[ib].Bx_fp_x, blocks[ib].By_fp_y,
                       blocks[ib].Bz_fp_z, blocks[ib].Tx_, blocks[ib].Ty_,
                       blocks[ib].Tz_, blocks[ib].fp_to_sp);
            mhd_compute_conservatives(blocks[ib].W_sp, blocks[ib].U_sp);
            blocks[ib].transform_sp_to_cv(blocks[ib].W_sp, blocks[ib].W_cv);
        }
    }

    void transfer_from_snapshot(std::map<BlockForest::BlockKey,int>& key_to_ib,
                                std::vector<BlockSnap>& snap){
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
                    restrict_snap_child(snap[sibs[s]], acc, cx, cy, cz);
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

    void adapt(){
        if constexpr (is_mhd){
            if(cfg.active[_z_]){
                if(Master)
                    std::cout<<std::endl<<"ERROR: dynamic AMR for 3D MHD is not implemented yet"
                               <<" (face-B prolongate is 2D-only)"<<std::endl;
                exit(1);
            }
        }
        std::vector<BlockSnap> snap(nblocks);
        for(int ib=0; ib<nblocks; ib++){
            snap[ib] = make_empty_snap(ib, "snap");
            capture_block_snap(ib, snap[ib]);
        }
        auto key_to_ib = snapshot_keys();

        std::vector<int> to_refine;
        std::vector<std::vector<int>> to_derefine;
        tag_blocks(forest, blocks, to_refine, to_derefine,
                   cfg.amr_max_level, cfg.amr_criterion);
        if(to_refine.empty() && to_derefine.empty()) return;

        int old_M = forest.max_level();
        auto deref_keys = forest.keys_of(to_derefine);
        if(!to_refine.empty()) forest.refine_blocks(to_refine);
        if(!deref_keys.empty()) forest.derefine_blocks_keys(deref_keys);
        forest.enforce_2to1_balance();

        build_block_solvers();
        transfer_from_snapshot(key_to_ib, snap);
        if constexpr (is_mhd){
            if(forest.max_level()==0) Sync_face_B_mhd();
            else Exchange_face_B_mhd();
        }
        for(int ib=0; ib<nblocks; ib++) finish_block_ic(ib);
        recompute_dt();
        if(forest.max_level() != old_M)
            init_W_glob(Xg, Yg, Zg, x_fp_);
    }
};

#endif  // MESH_HPP_
