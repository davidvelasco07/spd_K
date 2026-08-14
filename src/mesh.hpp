#ifndef MESH_HPP_
#define MESH_HPP_

#include <algorithm>
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

    //All blocks' evolution arrays live here, one allocation per array name
    //with a leading block axis, so a phase can be one kernel over the mesh
    //instead of one kernel per block.
    BlockPack pack;

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

    Block make_block(const dimension& Xdim, const dimension& Ydim, const dimension& Zdim,
                     int ib){
        if constexpr (is_hydro)
            return Hydro_ader(comm_,p_,Xdim,Ydim,Zdim,x_,w_,x_sp_,x_fp_,nu_,beta_,false,
                              &pack, ib);
        else
            return MHD_ader(comm_,p_,Xdim,Ydim,Zdim,x_,w_,x_sp_,x_fp_,false,
                            &pack, ib);
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
        //Drop the previous pack before the new one is sized: adapt() changes
        //the block count, so every array is reallocated with a new leading
        //extent and the old slices must not keep it alive.
        pack.reset(nblocks);
        for(int ib=0; ib<nblocks; ib++){
            const MeshBlock& b = forest.blocks[ib];
            Xd.emplace_back(x_dim_for_block(b, p_, x_fp_));
            Yd.emplace_back(y_dim_for_block(b, p_, x_fp_));
            Zd.emplace_back(z_dim_for_block(b, p_, x_fp_));
            blocks.push_back(make_block(Xd[ib], Yd[ib], Zd[ib], ib));
        }
        build_geometry_pack();
        build_pack_views();
        build_neighbor_tables();
        build_xchg_tables();
    }

    //Per-block geometry that batched kernels need by block index: element
    //size and the FV sub-grid coordinates. Blocks differ only in these, which
    //is exactly what lets one kernel span refinement levels.
    Vector hx_p, hy_p, hz_p;      //element size per block
    Matrix fvx_p, fvy_p, fvz_p;   //FV face coordinates per block

    //Whole-pack views of the arrays the batched phases touch, cached so the
    //hot loop does no map lookups. Rebuilt with the pack on every adapt.
    struct PackViews {
        SD_Solution U_ader_fp_x, U_ader_fp_y, U_ader_fp_z;
        SD_Solution F_ader_fp_x, F_ader_fp_y, F_ader_fp_z;
        SD_Solution U_sp, W_sp, W_cv, U_cv, U_ader_sp, U0_sp, T_sweep;
        SD_Solution T_fp_x, T_fp_y, T_fp_z;
        FV_Solution U_old, U_new, W_old, W_new, theta;
        FV_Solution F_x, F_y, F_z;
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
        pv.U0_sp       = sd_pack_view(pack,"U0_sp");
        pv.T_sweep     = sd_pack_view(pack,"T_sweep");
        pv.U_cv        = sd_pack_view(pack,"U_cv");
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

        auto pack_faces = [&](Matrix& M, const char* nm, std::vector<dimension>& D){
            int n = D[0].fv_nfaces;
            M = Matrix(nm, nblocks, n);
            Matrix_h h = setup_mirror(M);
            for(int b=0;b<nblocks;b++){
                Vector_h f = setup_mirror(D[b].fv_faces);
                setup_pull(D[b].fv_faces, f);
                for(int i=0;i<n;i++) h(b,i) = f(i);
            }
            setup_push(M,h);
        };
        pack_faces(fvx_p,"pack_fvx",Xd);
        pack_faces(fvy_p,"pack_fvy",Yd);
        pack_faces(fvz_p,"pack_fvz",Zd);
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

    //Per-block neighbour indices and side types on a uniform mesh, uploaded
    //once so the batched exchanges can read them inside the kernel instead of
    //the host re-deriving them per block per launch.
    IntVector nbrL_[3], nbrR_[3], typL_[3], typR_[3];

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

    static bool new_xchg(){
        static bool v = getenv("SPD_NEW_XCHG") != nullptr;
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
                if(new_xchg()){
                    gather_all_fp(dim);
                } else {
                    block_boundary_sd_b(P,nbrL_[dim],nbrR_[dim],typL_[dim],typR_[dim],dim);
                }
                Kokkos::deep_copy(packed, P.Vector);
                Kokkos::deep_copy(P.Vector, pre);
                forest_exchange_fp(forest, blocks, dim);
                report_exchange_diff(P, packed, dim);
            }
            return;   //the forest result is left in place
        }
        for(int dim=0; dim<3; dim++){
            if(!cfg.active[dim]) continue;
            if(new_xchg()){
                gather_all_fp(dim);
            } else if(forest.max_level()==0 && !no_pack()){
                block_boundary_sd_b(fp_pack(dim),nbrL_[dim],nbrR_[dim],
                                    typL_[dim],typR_[dim],dim);
            } else {
                forest_exchange_fp(forest, blocks, dim);
            }
        }
    }

    void Exchange_fv_field(FV_Solution Block::*member, FV_Solution* packed=nullptr){
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
                if(cfg.active[dim]) correct_coarse_fine_flux(forest, blocks, dim);
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
        if(cfg.muscl_only) return;
        compute_primitives(pv.U_new, pv.W_new);
        for(int b=0;b<nblocks;b++)
            detect_troubles(blocks[b].W_new,blocks[b].W_old,blocks[b].troubles,
                            blocks[b].cascade,
                            blocks[b].alpha_x,blocks[b].alpha_y,blocks[b].alpha_z,
                            Xd[b],Yd[b],Zd[b],1,(1<<_d_)|(1<<_p_));
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

    void FV_theta_batched(){
        if(cfg.muscl_only){
            Kokkos::deep_copy(pv.theta.Vector, 1.0);
            return;
        }
        for(int b=0;b<nblocks;b++) blocks[b].FV_theta();
    }

    void FV_Update_solution_hydro(){
        { Region r("FV_begin"); FV_begin_batched(); }
        for(int ader=0;ader<n_ader;ader++){
            { Region r("FV_flux_update"); FV_flux_update_batched(ader); }
            { Region r("Exchange_U_old"); Exchange_fv_field(&Block::U_old,&pv.U_old); }
            { Region r("Exchange_U_new"); Exchange_fv_field(&Block::U_new,&pv.U_new); }
            { Region r("FV_detect"); FV_detect_batched(); }
            if(!cfg.muscl_only){ Region r("Exchange_cascade");
                                 Exchange_fv_field(&Block::cascade); }
            { Region r("FV_theta"); FV_theta_batched(); }
            if(!cfg.muscl_only){ Region r("Exchange_theta");
                                 Exchange_fv_field(&Block::theta); }
            { Region r("FV_blend");
              for(int b=0;b<nblocks;b++)
                  blocks[b].FV_blend(ader,Xd[b],Yd[b],Zd[b]); }
            if(forest.max_level()>0){
                Region r("correct_cf_fv_flux");
                for(int dim=0; dim<3; dim++)
                    if(cfg.active[dim])
                        correct_coarse_fine_fv_flux(forest, blocks, dim);
            }
            { Region r("FV_commit"); FV_commit_batched(ader); }
        }
        { Region r("FV_end"); FV_end_batched(); }
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
        Region r("TaskCopyCons");
        sync_block_dt();
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
        if(cfg.integrator==_integrator_rk_ && d->rk_a[stage-1]>0){
            double a = d->rk_a[stage-1];
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
            for(int b=0;b<nblocks;b++)
                mhd_B_to_U(blocks[b].U_sp, blocks[b].Bx_fp_x, blocks[b].By_fp_y,
                           blocks[b].Bz_fp_z, blocks[b].Tx_, blocks[b].Ty_,
                           blocks[b].Tz_, blocks[b].fp_to_sp);
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskConsToPrim(Driver* d, int stage){
        Region r("TaskConsToPrim");
        if constexpr (is_hydro){
            compute_primitives(pv.U_sp, pv.W_sp);
            transform_sp_to_cv_batched(pv.W_sp, pv.W_cv);
        } else {
            for(int b=0;b<nblocks;b++){
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
        Region r("ComputeDt");
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
        double m_before = mass_dbg ? total_mass() : 0.0;
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
        if constexpr (is_mhd) report_divb("after transfer");
        if constexpr (is_mhd){
            if(forest.max_level()==0) Sync_face_B_mhd();
            else Exchange_face_B_mhd();
        }
        if constexpr (is_mhd) report_divb("after exchange");
        for(int ib=0; ib<nblocks; ib++) finish_block_ic(ib);
        if(mass_dbg && Master){
            double m_after = total_mass();
            std::cout<<std::endl<<"[adapt] step "<<this->n_step
                     <<" nblocks "<<nblocks
                     <<" mass "<<std::setprecision(17)<<m_before<<" -> "<<m_after
                     <<"  rel "<<std::setprecision(6)
                     <<(m_before!=0.0 ? (m_after-m_before)/m_before : 0.0)<<std::endl;
        }
        recompute_dt();
        if(forest.max_level() != old_M)
            init_W_glob(Xg, Yg, Zg, x_fp_);
    }
};

#endif  // MESH_HPP_
