#ifndef DRIVER_HPP_
#define DRIVER_HPP_
//========================================================================================
// spd_K driver
//
// Owns the time-integration loop and the named task-list phases that physics modules
// register into (athenak-style). A physics module derives from PhysicsModule and, in
// AssembleTasks(), pushes its per-stage work into the shared task lists. The driver
// runs the SSP-RK stage loop (or a single ADER stage), reduces/advances dt, handles
// the tlim clamping and the output cadence.
//
// Phases (athenak naming):
//   before_timeintegrator : once per cycle, before the stage loop (e.g. RK save-state)
//   before_stagen         : start of each stage (reserved; e.g. gravity Solve hook)
//   stagen                : the fluid/field update for the stage
//   after_stagen          : end of each stage (reserved)
//   after_timeintegrator  : once per cycle, after the stage loop (e.g. cons->prim)
//   after_cycle           : once per cycle, after t/n_step advance and before ComputeDt
//                           (AMR adapt registers here)
//
// A self-gravity multigrid Solve (your ongoing work) slots in between before_stagen
// and stagen without touching the fluid task graph; AMR stays in after_cycle.
//========================================================================================

#include <map>
#include <memory>
#include <string>
#include "tasklist/task_list.hpp"

class Driver;

//----------------------------------------------------------------------------------------
//! \class PhysicsModule
//  \brief base class for a physics package that plugs into the Driver's task lists.

class PhysicsModule {
 public:
  virtual ~PhysicsModule() = default;

  // Shared time state. The Driver is the loop authority: it advances t, counts steps,
  // and writes dt each cycle; the module's numeric kernels read the inherited dt.
  double t  = 0.0;
  double dt = 0.0;
  double Dt = 0.0;   // step size set at construction (first cycle)
  int n_step   = 0;
  int n_output = 0;

  // Register this module's tasks into the driver's task-list phases.
  virtual void AssembleTasks(Driver *d) = 0;
  // CFL-limited step size from the current state.
  virtual double ComputeDt() = 0;
  // Write outputs for the current state.
  virtual void WriteOutputs() = 0;
};

//----------------------------------------------------------------------------------------
//! \class Driver
//  \brief owns the task-list phases and drives the time-integration loop.

class Driver {
 public:
  std::map<std::string, std::shared_ptr<TaskList>> tl_map;
  PhysicsModule *pmod = nullptr;

  // Integrator configuration (owned here, not in the physics module).
  int integrator;
  int n_stages;          // RK stages (1 for ADER)
  int rk_order = 0;
  double rk_a[3] = {0.0, 0.0, 0.0};   // per-stage convex weights of the SSP combination

  explicit Driver(PhysicsModule *mod) : pmod(mod) {
    integrator = cfg.integrator;
    if (integrator == _integrator_rk_) {
      rk_order = cfg.rk_order;
      n_stages = ssp_rk_coefficients(cfg.rk_order, rk_a);
    } else {
      n_stages = 1;
    }
    const char *phases[] = {"before_timeintegrator", "before_stagen", "stagen",
                            "after_stagen", "after_timeintegrator", "after_cycle"};
    for (auto name : phases)
      tl_map[name] = std::make_shared<TaskList>();
    pmod->AssembleTasks(this);
  }

  // Run one task-list phase to completion. Synchronous today (one pass suffices);
  // the retry loop is kept so async receives can return TaskStatus::incomplete later.
  void ExecuteTaskList(const std::string &name, int stage) {
    auto it = tl_map.find(name);
    if (it == tl_map.end() || it->second->Empty()) return;
    auto tl = it->second;
    tl->Reset();
    int guard = 0;
    while (tl->DoAvailable(this, stage) != TaskListStatus::complete) {
      if (++guard > 1000000) {
        if (Master) std::cout << "ERROR: task list '" << name << "' is stuck" << std::endl;
        break;
      }
    }
  }

  void Execute(double t_end, double dt_output) {
    pmod->dt = pmod->Dt;
    double t_output = dt_output;
    //SPD_DT_TRACE=1 prints (step, t, dt) every step. dt collapsing while
    //staying finite is invisible otherwise: the non-finite-dt guard never
    //fires and the run grinds instead of failing, so the only symptom is a
    //step count that stops scaling with tlim.
    const bool dt_trace = getenv("SPD_DT_TRACE") != nullptr;
    const double dt0 = pmod->dt;
    //An output interval finer than the timestep cannot be honoured by writing
    //more often -- there is nothing in between two steps. Say so once and then
    //write every step, rather than shrinking dt to match (see below).
    const bool sub_cycle = (dt_output > 0.0 && dt_output < dt0);
    if (sub_cycle && Master)
      std::cout << "WARNING: output/dt = " << dt_output << " is smaller than the"
                << " timestep " << dt0 << "; writing every step instead."
                << " The timestep is NOT reduced to match." << std::endl;

    // Time only the evolution loop; subtract time spent writing outputs.
    Kokkos::fence();
    Kokkos::Timer timer;
    double t_io = 0;
    int step0 = pmod->n_step;

    while (pmod->t < t_end) {
      ExecuteTaskList("before_timeintegrator", 0);
      for (int s = 1; s <= n_stages; s++) {
        ExecuteTaskList("before_stagen", s);
        ExecuteTaskList("stagen", s);
        ExecuteTaskList("after_stagen", s);
      }
      ExecuteTaskList("after_timeintegrator", 1);

      pmod->t += pmod->dt;
      pmod->n_step++;
      ExecuteTaskList("after_cycle", 1);
      pmod->dt = pmod->ComputeDt();

      if (dt_trace && Master)
        std::cout << "[dt] step " << (pmod->n_step - step0)
                  << " t=" << pmod->t << " dt=" << pmod->dt
                  << " dt/dt0=" << (dt0 != 0.0 ? pmod->dt/dt0 : 0.0)
                  << std::endl;
      //A collapsing-but-finite dt means the run never ends. Fail loudly
      //instead: this is a bug every time, not a physical regime.
      if (dt0 > 0.0 && pmod->dt > 0.0 && pmod->dt < 1e-6*dt0) {
        if (Master)
          std::cout << std::endl << "ERROR: dt collapsed to " << pmod->dt
                    << " (" << pmod->dt/dt0 << " of its initial value) at step "
                    << (pmod->n_step - step0) << ", t = " << pmod->t
                    << "; run would not finish. Set SPD_DT_TRACE=1 to see the"
                    << " approach." << std::endl;
        break;
      }
      //time/nlim caps the step count, so a throughput measurement is bounded
      //by steps rather than by an end time it may never reach.
      if (cfg.nlim > 0 && (pmod->n_step - step0) >= cfg.nlim) break;

      if (Master) std::cout << ".";
      if (cfg.outputs) {
        if (pmod->t >= t_output) {
          //Anchor the schedule to multiples of dt_output rather than to the
          //time we happened to reach: `t_output = t + dt_output` lets the
          //cadence drift by up to one timestep per output.
          do { t_output += dt_output; } while (t_output <= pmod->t);
          Kokkos::fence();
          Kokkos::Timer io_timer;
          pmod->WriteOutputs();
          t_io += io_timer.seconds();
        }
        //Land exactly on the next output time -- but ONLY as a truncation of a
        //CFL-chosen step, never as a way to make the step smaller than the
        //scheme asked for. Without the `sub_cycle` guard an output/dt below the
        //timestep sets dt = dt_output EVERY step: the run then takes
        //t_end/dt_output steps and writes a dump on each one. Measured the
        //expensive way -- `output/dt=1e-9 tlim=1.0` wrote 740 GB and filled the
        //filesystem, which then broke an unrelated build (nvcc could not open a
        //temporary). The cadence must never drive the timestep.
        if (!sub_cycle && pmod->t + pmod->dt > t_output)
          pmod->dt = t_output - pmod->t;
      }
    }
    Kokkos::fence();
    double t_evol = timer.seconds() - t_io;
    if (Master)
      std::cout << std::endl << "evolution: " << (pmod->n_step - step0)
                << " steps, " << t_evol << " s" << std::endl;
  }
};

#endif  // DRIVER_HPP_
