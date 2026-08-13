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
//
// A self-gravity multigrid Solve (your ongoing work) slots in between before_stagen
// and stagen without touching the fluid task graph; AMR (later) stays post-cycle,
// outside the stage lists.
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
                            "after_stagen", "after_timeintegrator"};
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
      pmod->dt = pmod->ComputeDt();

      if (Master) std::cout << ".";
      if (cfg.outputs) {
        if (pmod->t >= t_output) {
          t_output = pmod->t + dt_output;
          Kokkos::fence();
          Kokkos::Timer io_timer;
          pmod->WriteOutputs();
          t_io += io_timer.seconds();
        }
        if (pmod->t + pmod->dt > t_output)
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
