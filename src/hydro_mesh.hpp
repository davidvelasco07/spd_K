using namespace std;

//Multi-block (unigrid) hydro driver: the rank domain is split into a
//regular grid of meshblocks, each one a full Hydro_ader solver over its own
//sub-domain (own arrays, own ghost elements). The driver sequences the
//per-block phases and replaces every halo exchange of the single-block path
//with block-to-block ghost copies, so a multi-block run reproduces the
//single-block solution to round-off (the per-cell arithmetic is identical).
//
//This is the foundation for AMR: blocks are the refinement unit, and the
//exchange points marked here are where prolongation/restriction (amr.cpp)
//will act once neighbors can differ by one level.
struct Hydro_mesh{
    int nbx, nby, nbz; //blocks per direction (this rank)
    int nblocks;
    int NBx, NBy, NBz; //elements per block per direction
    int n_ader;
    int n_stages;
    double rk_a[3];

    double t=0;
    double dt;
    double Dt;
    int n_step=0;
    int n_output=0;

    vector<Hydro_ader> blocks;
    vector<dimension> Xd, Yd, Zd; //per-block dimensions (coordinates)
    dimension Xg, Yg, Zg;         //rank-wide dimensions (stitched output)
    SD_Solution W_glob;           //stitched W_cv for outputs

    Hydro_mesh(
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
    ) : Xg(X_dim), Yg(Y_dim), Zg(Z_dim)
    {
        NBx=_NBx; NBy=_NBy; NBz=_NBz;
        nbx = X_dim.N/NBx;
        nby = Y_dim.N/NBy;
        nbz = Z_dim.N/NBz;
        nblocks = nbx*nby*nbz;

        for(int bz=0; bz<nbz; bz++)
        for(int by=0; by<nby; by++)
        for(int bx=0; bx<nbx; bx++){
            Xd.emplace_back(_x_,X_dim.N_global,NBx,cfg.active[_x_] ? p:0,comm.x*X_dim.N+bx*NBx,X_dim.L,x_fp,cfg.active[_x_]);
            Yd.emplace_back(_y_,Y_dim.N_global,NBy,cfg.active[_y_] ? p:0,comm.y*Y_dim.N+by*NBy,Y_dim.L,x_fp,cfg.active[_y_]);
            Zd.emplace_back(_z_,Z_dim.N_global,NBz,cfg.active[_z_] ? p:0,comm.z*Z_dim.N+bz*NBz,Z_dim.L,x_fp,cfg.active[_z_]);
            int b = blocks.size();
            blocks.emplace_back(comm,p,Xd[b],Yd[b],Zd[b],x,w,x_sp,x_fp,nu,beta,false);
        }
        n_ader   = blocks[0].n_ader;
        n_stages = blocks[0].n_stages;
        for(int s=0;s<3;s++) rk_a[s]=blocks[0].rk_a[s];

        Dt = blocks[0].Dt;
        for(int b=1;b<nblocks;b++)
            Dt = min(Dt,blocks[b].Dt);

        W_glob.init("W_cv",1,blocks[0].nvar,Zg,Yg,Xg,0,0,0);

        if(Master)
            cout<<"meshblocks = ("<<nbx<<","<<nby<<","<<nbz<<") of ("
                <<NBx<<","<<NBy<<","<<NBz<<") elements"<<endl
                <<"dx = "<<Xg.h<<" dt = "<<Dt<<endl;
        Write_outputs();
    }

    //Block index helpers (row-major x-fastest, periodic wrap)
    int bid(int bx, int by, int bz){
        return bx + nbx*(by + nby*bz);
    }

    void neighbors(int b, int dim, int &L, int &R, int &tL, int &tR){
        int c[3]  = {b%nbx, (b/nbx)%nby, b/(nbx*nby)};
        int nb[3] = {nbx, nby, nbz};
        int cl[3] = {c[0],c[1],c[2]};
        int cr[3] = {c[0],c[1],c[2]};
        cl[dim] = (c[dim]-1+nb[dim])%nb[dim];
        cr[dim] = (c[dim]+1)%nb[dim];
        L = bid(cl[0],cl[1],cl[2]);
        R = bid(cr[0],cr[1],cr[2]);
        //Interior faces always copy from the neighbor; a domain edge keeps
        //the physical boundary type (periodic edges wrap to the far block)
        tL = (c[dim]==0          && cfg.bc[dim]==_gradfree_) ? _gradfree_ : _periodic_;
        tR = (c[dim]==nb[dim]-1  && cfg.bc[dim]==_gradfree_) ? _gradfree_ : _periodic_;
    }

    SD_Solution& fp(int b, int dim){
        return dim==_x_ ? blocks[b].U_ader_fp_x
             : dim==_y_ ? blocks[b].U_ader_fp_y
                        : blocks[b].U_ader_fp_z;
    }

    //Interface ghost exchange of the flux-point arrays (each direction only
    //needs its own staggered array, as in the single-block halo exchange)
    void Exchange_fp(){
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            for(int b=0; b<nblocks; b++){
                int L,R,tL,tR;
                neighbors(b,dim,L,R,tL,tR);
                block_boundary_sd(fp(b,dim),fp(L,dim),fp(R,dim),tL,tR,dim);
            }
        }
    }

    //Ghost-cell exchange of one FV array selected per block by `get`.
    //Directions run sequentially over all blocks so corner ghosts propagate
    //(the y sweep reads neighbor rows whose x ghosts were just filled),
    //matching the single-block sequential-direction exchange.
    template<class Get>
    void Exchange_fv(Get get){
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            for(int b=0; b<nblocks; b++){
                int L,R,tL,tR;
                neighbors(b,dim,L,R,tL,tR);
                block_boundary_fv(get(blocks[b]),get(blocks[L]),get(blocks[R]),tL,tR,dim);
            }
        }
    }

    //One evaluation of the spatial operator on all blocks (the multi-block
    //version of Hydro_ader::Solve_fluxes)
    void Solve_fluxes(){
        for(int b=0;b<nblocks;b++)
            blocks[b].Fluxes_pre();
        Exchange_fp();
        for(int b=0;b<nblocks;b++)
            blocks[b].Riemann_Solver();

        #ifdef VISCOSITY
        for(int b=0;b<nblocks;b++)
            blocks[b].Viscosity(Xg.h,Yg.h,Zg.h);
        Exchange_fp();
        for(int b=0;b<nblocks;b++)
            blocks[b].Rusanov_Solver();
        #endif
    }

    //Multi-block version of Hydro_ader::FV_Update_solution: same phase
    //sequence, with block-to-block exchanges at the same points
    void FV_Update_solution(){
        for(int b=0;b<nblocks;b++)
            blocks[b].FV_begin();
        for(int ader=0;ader<n_ader;ader++){
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_flux_update(ader,Xd[b],Yd[b],Zd[b]);
            Exchange_fv([](Hydro_ader &blk)->FV_Solution&{return blk.U_old;});
            Exchange_fv([](Hydro_ader &blk)->FV_Solution&{return blk.U_new;});
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_detect(Xd[b],Yd[b],Zd[b]);
            Exchange_fv([](Hydro_ader &blk)->FV_Solution&{return blk.troubles;});
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_theta();
            Exchange_fv([](Hydro_ader &blk)->FV_Solution&{return blk.theta;});
            for(int b=0;b<nblocks;b++)
                blocks[b].FV_apply(ader,Xd[b],Yd[b],Zd[b]);
        }
        for(int b=0;b<nblocks;b++)
            blocks[b].FV_end();
    }

    void Update_solution(){
        if(cfg.fallback)
            FV_Update_solution();
        else
            for(int b=0;b<nblocks;b++)
                blocks[b].Update_solution(Xg.h,Yg.h,Zg.h);
    }

    void ADER_step(){
        for(int b=0;b<nblocks;b++)
            blocks[b].copy_ader(blocks[b].U_sp,blocks[b].U_ader_sp);
        for(int ader=0;ader<n_ader;ader++){
            Solve_fluxes();
            if(ader<n_ader-1)
                for(int b=0;b<nblocks;b++)
                    blocks[b].Update_prediction(Xg.h,Yg.h,Zg.h);
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

    void time_evolution(CommHelper comm, double t_end, double dt_output){
        dt=Dt;
        double t_output=dt_output;

        while(t<t_end){
            for(int b=0;b<nblocks;b++)
                blocks[b].dt = dt;
            if(cfg.integrator==_integrator_rk_)
                RK_step();
            else
                ADER_step();
            double dt_next=1e300;
            for(int b=0;b<nblocks;b++){
                compute_primitives(blocks[b].U_sp,blocks[b].W_sp);
                blocks[b].transform_sp_to_cv(blocks[b].W_sp,blocks[b].W_cv);
                dt_next=min(dt_next,compute_dt(blocks[b].W_cv,Xg.h,Yg.h,Zg.h));
            }
            t+=dt;
            n_step++;
            dt=dt_next;

            //Outputs
            if(Master) cout<<".";
            if(t>=t_output){
                t_output=t+dt_output;
                Write_outputs();
            }
            if(t+dt>t_output){
                dt=t_output-t;
            }
        }
        cout<<endl;
    }

    void Write_outputs(){
        if(Master)
            cout<<endl<<"OUTPUT "<<n_output<<endl;
        for(int b=0;b<nblocks;b++){
            int bx=b%nbx, by=(b/nbx)%nby, bz=b/(nbx*nby);
            gather_block(blocks[b].W_cv,W_glob,bx*NBx,by*NBy,bz*NBz);
        }
        Write(W_glob,n_output++);
    }
};
