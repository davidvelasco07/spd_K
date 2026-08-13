using namespace std;

#include <map>

#include "forest.hpp"
#include "amr_criteria.hpp"

//Block-forest hydro driver: uniform multiblock (single level) and mixed-level AMR.
struct Hydro_mesh{
    BlockForest forest;
    int NBx, NBy, NBz;
    int nblocks;
    int n_ader;
    int n_stages;
    double rk_a[3];

    double t=0;
    double dt;
    double Dt;
    int n_step=0;
    int n_output=0;

    vector<Hydro_ader> blocks;
    vector<dimension> Xd, Yd, Zd;
    dimension Xg, Yg, Zg;
    SD_Solution W_glob;

    Hydro_mesh(
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
        double nu,
        double beta
    ) : forest(std::move(f)), Xg(X_dim), Yg(Y_dim), Zg(Z_dim)
    {
        NBx=_NBx; NBy=_NBy; NBz=_NBz;
        nblocks = forest.Nblocks();
        if(forest.max_level()>0 && cfg.integrator==_integrator_ader_){
            if(Master) cout<<"ERROR: ADER is not supported with mixed-level AMR (use rk2/rk3)"<<endl;
            exit(1);
        }
        build_block_solvers(comm, p, x, w, x_sp, x_fp, nu, beta);

        n_ader   = blocks[0].n_ader;
        n_stages = blocks[0].n_stages;
        for(int s=0;s<3;s++) rk_a[s]=blocks[0].rk_a[s];

        Dt = blocks[0].Dt;
        for(int b=1;b<nblocks;b++)
            Dt = min(Dt, blocks[b].Dt);

        init_W_glob(X_dim, Y_dim, Z_dim, x_fp);
        if(Master)
            cout<<"forest blocks = "<<nblocks<<" max_level = "<<forest.max_level()
                <<" NB = ("<<NBx<<","<<NBy<<","<<NBz<<") dt = "<<Dt<<endl;
        Write_outputs();
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

    void build_block_solvers(CommHelper comm, int p, double* x, double* w,
                             double* x_sp, double* x_fp, double nu, double beta){
        blocks.clear(); Xd.clear(); Yd.clear(); Zd.clear();
        nblocks = forest.Nblocks();
        for(int ib=0; ib<nblocks; ib++){
            const MeshBlock& b = forest.blocks[ib];
            Xd.emplace_back(x_dim_for_block(b, p, x_fp));
            Yd.emplace_back(y_dim_for_block(b, p, x_fp));
            Zd.emplace_back(z_dim_for_block(b, p, x_fp));
            blocks.emplace_back(comm,p,Xd[ib],Yd[ib],Zd[ib],x,w,x_sp,x_fp,nu,beta,false);
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
            //Stitched output lives on the finest level, whose cell faces are
            //not the base grid's. Dump them so post-processing integrates with
            //the real (non-uniform, flux-point) cell widths.
            Write_dimensions(Xf, Yf, Zf);
        } else {
            W_glob.init("W_cv",1,blocks[0].nvar,Z_dim,Y_dim,X_dim,0,0,0);
        }
    }

    void recompute_dt(){
        Dt = 1e300;
        for(int b=0;b<nblocks;b++){
            compute_primitives(blocks[b].U_sp, blocks[b].W_sp);
            blocks[b].transform_sp_to_cv(blocks[b].W_sp, blocks[b].W_cv);
            Dt = min(Dt, compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h));
        }
    }

    SD_Solution& fp(int b, int dim){
        return dim==_x_ ? blocks[b].U_ader_fp_x
             : dim==_y_ ? blocks[b].U_ader_fp_y
                        : blocks[b].U_ader_fp_z;
    }

    SD_Solution& Ffp(int b, int dim){
        return dim==_x_ ? blocks[b].F_ader_fp_x
             : dim==_y_ ? blocks[b].F_ader_fp_y
                        : blocks[b].F_ader_fp_z;
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

    void Exchange_fv_field(FV_Solution Hydro_ader::*member){
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
        //A coarse->fine injection writes a transverse ghost row across the
        //block's whole extent, prolongating one coarse neighbour even into
        //the columns that lie outside this block. Those corner cells then
        //disagree with the same-level neighbour's view of the same cell, and
        //since the reconstruction is multidimensional the two blocks compute
        //different fluxes on their shared face. Replaying the earlier
        //directions refills those columns from the neighbour that owns them.
        if(forest.max_level()>0)
            for(int dim=0; dim<3; dim++)
                if(cfg.active[dim])
                    forest_exchange_fv_same(forest, blocks, member, dim);
    }

    void Solve_fluxes(){
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

    void FV_Update_solution(){
        for(int b=0;b<nblocks;b++) blocks[b].FV_begin();
        for(int ader=0;ader<n_ader;ader++){
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_flux_update(ader,Xd[b],Yd[b],Zd[b]);
            Exchange_fv_field(&Hydro_ader::U_old);
            Exchange_fv_field(&Hydro_ader::U_new);
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_detect(Xd[b],Yd[b],Zd[b]);
            if(!cfg.muscl_only) Exchange_fv_field(&Hydro_ader::troubles);
            for(int b=0;b<nblocks;b++) blocks[b].FV_theta();
            if(!cfg.muscl_only) Exchange_fv_field(&Hydro_ader::theta);
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

    void Update_solution(){
        if(cfg.fallback) FV_Update_solution();
        else for(int b=0;b<nblocks;b++)
            blocks[b].Update_solution(Xd[b].h,Yd[b].h,Zd[b].h);
    }

    void ADER_step(){
        for(int b=0;b<nblocks;b++)
            blocks[b].copy_ader(blocks[b].U_sp,blocks[b].U_ader_sp);
        for(int ader=0;ader<n_ader;ader++){
            Solve_fluxes();
            if(ader<n_ader-1)
                for(int b=0;b<nblocks;b++)
                    blocks[b].Update_prediction(Xd[b].h,Yd[b].h,Zd[b].h);
        }
        Update_solution();
    }

    void RK_step(){
        for(int b=0;b<nblocks;b++)
            Kokkos::deep_copy(blocks[b].U0_sp.Vector,blocks[b].U_sp.Vector);
        for(int s=0;s<n_stages;s++){
            for(int b=0;b<nblocks;b++)
                blocks[b].copy_ader(blocks[b].U_sp,blocks[b].U_ader_sp);
            Solve_fluxes();
            Update_solution();
            if(rk_a[s]>0)
                for(int b=0;b<nblocks;b++)
                    combine_solution(blocks[b].U_sp,blocks[b].U0_sp,rk_a[s]);
        }
    }

    void prolongate_to_finest(int ib, SD_Solution W, SD_Solution& out,
                              dimension& Xd, dimension& Yd, dimension& Zd,
                              int ox, int oy, int oz, int steps){
        if(steps==0){
            gather_block(W, out, ox, oy, oz);
            return;
        }
        int nchild = 1;
        for(int d=0; d<3; d++) if(cfg.active[d]) nchild *= 2;
        for(int c=0; c<nchild; c++){
            int cx=c&1, cy=(c>>1)&1, cz=(c>>2)&1;
            SD_Solution child("child",1,W.n_var,Zd,Yd,Xd,0,0,0);
            prolongate_block(W, child, amr_P, cx, cy, cz);
            prolongate_to_finest(ib, child, out, Xd, Yd, Zd,
                ox*2+cx*NBx, oy*2+cy*NBy, oz*2+cz*NBz, steps-1);
        }
    }

    //The conserved integral, summed block by block over each block's own
    //cells. The stitched dump interpolates coarse blocks up onto the finest
    //grid, and interpolation need not preserve an integral, so on a
    //mixed-level mesh the dump cannot be used to check conservation.
    double total_mass(){
        double M=0;
        for(int b=0;b<nblocks;b++)
            M += blocks[b].fv_mass(blocks[b].W_cv, Xd[b], Yd[b], Zd[b]);
        return M;
    }

    void Write_outputs(){
        if(Master) cout<<endl<<"OUTPUT "<<n_output<<endl;
        if(Master){
            std::ofstream f(output_folder()+"mass.txt",
                            n_output==0 ? std::ios::trunc : std::ios::app);
            f<<std::setprecision(17)<<t<<" "<<total_mass()<<endl;
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
        Write(W_glob,n_output);
        Write_amr_blocks(forest, n_output, NBx, NBy, NBz);
        n_output++;
    }

    map<BlockForest::BlockKey,int> snapshot_keys(){
        map<BlockForest::BlockKey,int> m;
        for(int ib=0; ib<nblocks; ib++) m[forest.block_key(ib)] = ib;
        return m;
    }

    void finish_block_ic(int ib){
        compute_primitives(blocks[ib].U_sp, blocks[ib].W_sp);
        blocks[ib].transform_sp_to_cv(blocks[ib].W_sp, blocks[ib].W_cv);
    }

    void transfer_from_snapshot(map<BlockForest::BlockKey,int>& key_to_ib,
                                vector<SD_Solution>& snap){
        //A block can gain more than one level in a single adapt: 2:1 balancing
        //refines blocks that tagging has already refined, so with max_level>1
        //a new block's nearest snapshot ancestor can be two or more levels
        //coarser. Walk up until an ancestor is found and record the child
        //offset taken at each level, so the descent can prolongate once per
        //level instead of failing to match and leaving the block unwritten.
        auto find_ancestor = [&](BlockForest::BlockKey key,
                                 vector<array<int,3>>& chain)->int{
            BlockForest::BlockKey k = key;
            while(get<0>(k) > 0){
                chain.push_back({(int)(get<1>(k)%2),
                                 (int)(get<2>(k)%2),
                                 (int)(get<3>(k)%2)});
                k = {get<0>(k)-1, get<1>(k)/2, get<2>(k)/2, get<3>(k)/2};
                auto it = key_to_ib.find(k);
                if(it != key_to_ib.end()) return it->second;
            }
            return -1;
        };

        for(int ib=0; ib<nblocks; ib++){
            BlockForest::BlockKey key = forest.block_key(ib);
            auto it = key_to_ib.find(key);
            if(it != key_to_ib.end()){
                Kokkos::deep_copy(blocks[ib].U_sp.Vector, snap[it->second].Vector);
                finish_block_ic(ib);
                continue;
            }
            vector<array<int,3>> chain;
            int pib = find_ancestor(key, chain);
            if(pib >= 0){
                //Views are reference counted, so `cur` keeps each intermediate
                //alive after the loop body ends.
                SD_Solution cur = snap[pib];
                for(int s=(int)chain.size()-1; s>=1; s--){
                    SD_Solution tmp("pro",1,blocks[ib].nvar,Zd[ib],Yd[ib],Xd[ib],0,0,0);
                    prolongate_block(cur, tmp, amr_P, chain[s][0], chain[s][1], chain[s][2]);
                    cur = tmp;
                }
                prolongate_block(cur, blocks[ib].U_sp, amr_P,
                                 chain[0][0], chain[0][1], chain[0][2]);
                finish_block_ic(ib);
                continue;
            }
            int n_sib = 1;
            for(int d=0; d<3; d++) if(forest.active[d]) n_sib *= 2;
            vector<int> sibs;
            for(int s=0; s<n_sib; s++){
                int cx=s&1, cy=(s>>1)&1, cz=(s>>2)&1;
                BlockForest::BlockKey ckey = {
                    get<0>(key)+1,
                    2*get<1>(key)+cx,
                    2*get<2>(key)+cy,
                    2*get<3>(key)+cz};
                auto cit = key_to_ib.find(ckey);
                if(cit != key_to_ib.end()) sibs.push_back(cit->second);
            }
            if((int)sibs.size()==n_sib){
                SD_Solution acc("acc",1,blocks[ib].nvar,Zd[ib],Yd[ib],Xd[ib],0,0,0);
                for(int s=0; s<n_sib; s++){
                    int cx=s&1, cy=(s>>1)&1, cz=(s>>2)&1;
                    restrict_block(snap[sibs[s]], acc, amr_RF, cx, cy, cz);
                }
                Kokkos::deep_copy(blocks[ib].U_sp.Vector, acc.Vector);
                finish_block_ic(ib);
                continue;
            }
            //Falling through would leave this block holding whatever its
            //freshly allocated views came with, which shows up later as a NaN
            //far from its cause. Refuse to continue on an unwritten block.
            if(Master)
                cout<<endl<<"ERROR: adapt could not transfer block "<<ib
                    <<" (level "<<get<0>(key)<<", logical "<<get<1>(key)<<","
                    <<get<2>(key)<<","<<get<3>(key)<<"): no snapshot ancestor "
                    <<"and only "<<sibs.size()<<" of "<<n_sib<<" children"<<endl;
            Kokkos::finalize();
            exit(1);
        }
    }

    void adapt(CommHelper comm, int p, double* x, double* w,
               double* x_sp, double* x_fp, double nu, double beta,
               dimension X_dim, dimension Y_dim, dimension Z_dim){
        vector<SD_Solution> snap(nblocks);
        for(int ib=0; ib<nblocks; ib++){
            snap[ib].init("snap", blocks[ib].n_ader, blocks[ib].nvar,
                          Zd[ib], Yd[ib], Xd[ib], 0, 0, 0);
            Kokkos::deep_copy(snap[ib].Vector, blocks[ib].U_sp.Vector);
        }
        auto key_to_ib = snapshot_keys();

        vector<int> to_refine;
        vector<vector<int>> to_derefine;
        tag_blocks(forest, blocks, to_refine, to_derefine,
                   cfg.amr_max_level, cfg.amr_criterion);
        if(to_refine.empty() && to_derefine.empty()) return;

        int old_M = forest.max_level();
        //Both lists index the pre-adapt block list, and refine_blocks erases
        //and appends blocks. Resolve the derefine groups to stable keys before
        //that happens, or they point at unrelated blocks afterwards and fuse
        //non-siblings into an oversized parent.
        auto deref_keys = forest.keys_of(to_derefine);
        if(!to_refine.empty()) forest.refine_blocks(to_refine);
        if(!deref_keys.empty()) forest.derefine_blocks_keys(deref_keys);
        forest.enforce_2to1_balance();

        build_block_solvers(comm, p, x, w, x_sp, x_fp, nu, beta);
        transfer_from_snapshot(key_to_ib, snap);
        recompute_dt();
        if(forest.max_level() != old_M)
            init_W_glob(X_dim, Y_dim, Z_dim, x_fp);
    }

    void time_evolution(
        CommHelper comm,
        double t_end,
        double dt_output,
        int p,
        double* x,
        double* w,
        double* x_sp,
        double* x_fp,
        double nu,
        double beta,
        dimension X_dim,
        dimension Y_dim,
        dimension Z_dim
    ){
        dt=Dt;
        double t_output=dt_output;

        while(t<t_end){
            for(int b=0;b<nblocks;b++) blocks[b].dt = dt;
            if(cfg.integrator==_integrator_rk_)
                RK_step();
            else
                ADER_step();

            for(int b=0;b<nblocks;b++){
                compute_primitives(blocks[b].U_sp,blocks[b].W_sp);
                blocks[b].transform_sp_to_cv(blocks[b].W_sp,blocks[b].W_cv);
            }
            //Each block's dt is tested on its own: min() propagates the first
            //argument when the second is NaN, so reducing first would hide a
            //diverged block behind a healthy one and let the run finish
            //"successfully" on garbage.
            Dt = 1e300;
            bool diverged = false;
            for(int b=0;b<nblocks;b++){
                double db = compute_dt(blocks[b].W_cv, Xd[b].h, Yd[b].h, Zd[b].h);
                if(!std::isfinite(db)) diverged = true;
                Dt = min(Dt, db);
            }
            if(diverged || !std::isfinite(Dt)){
                if(Master)
                    cout<<endl<<"ERROR: non-finite dt at step "<<n_step
                        <<" (t = "<<t<<"), solution has diverged"<<endl;
                Kokkos::finalize();
                exit(1);
            }

            t+=dt;
            n_step++;
            dt=Dt;

            if(cfg.adapt_interval>0 && n_step%cfg.adapt_interval==0){
                adapt(comm, p, x, w, x_sp, x_fp, nu, beta, X_dim, Y_dim, Z_dim);
                //Refinement halves h on the tagged blocks, so adapt() has
                //already recomputed Dt against the new mesh. Without picking
                //it up here the next step would run the pre-refinement dt at
                //double the CFL limit, on exactly the blocks that just gained
                //a coarse-fine interface.
                dt = Dt;
            }

            if(Master) cout<<".";
            if(t>=t_output){
                t_output=t+dt_output;
                Write_outputs();
            }
            if(t+dt>t_output) dt=t_output-t;
        }
        cout<<endl;
    }
};
