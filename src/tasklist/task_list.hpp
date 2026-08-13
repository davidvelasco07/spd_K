#ifndef TASKLIST_TASK_LIST_HPP_
#define TASKLIST_TASK_LIST_HPP_
//========================================================================================
// spd_K task list
//
// Ported (near-verbatim) from AthenaK's src/tasklist/task_list.hpp
//   AthenaXXX astrophysical plasma code
//   Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
//   Licensed under the 3-clause BSD License (the "LICENSE")
//
// Provides functionality to control (possibly asynchronous) execution using tasks.
// A physics module registers its per-stage work as Tasks with explicit dependencies;
// the Driver cycles the lists once per stage. The initial spd_K port keeps the
// boundary exchanges synchronous (every task returns TaskStatus::complete), but the
// TaskStatus::incomplete + DoAvailable retry machinery is preserved so async MPI
// receives can be added later without changing the task graph.
//
// The Task function signature is TaskStatus(Driver*, int stage).
//========================================================================================

#include <iostream>
#include <bitset>
#include <functional>
#include <list>

class Driver;

// Maximum size of a task list
#define NUMBER_TASKID_BITS 64

// return codes for functions working on individual Tasks and TaskLists
enum class TaskStatus {fail, complete, incomplete};
enum class TaskListStatus {running, stuck, complete, nothing_to_do};

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief container class for bit fields (used to encode Task IDs) and access functions

class TaskID {
 public:
  TaskID() = default;
  // ctor, default id = 0.
  explicit TaskID(unsigned int id) {
    if (id == 0) {
      bitfld_.reset();      // set all bits to zero
    } else {
      bitfld_.set((--id));  // set [id-1] bit to one
    }
  }

  void Clear() { bitfld_.reset(); }  // set all bits to zero
  // return true if input dependencies are clear
  bool CheckDependencies(const TaskID &dep) const {
    return ((bitfld_ & dep.bitfld_) == dep.bitfld_);
  }
  void PrintID() {std::cout << "TaskID = " << bitfld_.to_string() << std::endl;}
  // mark task with input TaskID as complete
  void SetComplete(const TaskID &rhs) { bitfld_ |= rhs.bitfld_; }

  bool operator== (const TaskID &rhs) const {return (bitfld_ == rhs.bitfld_); }
  bool operator!= (const TaskID &rhs) const {return (bitfld_ != rhs.bitfld_); }
  TaskID operator| (const TaskID &rhs) const {
    TaskID ret;
    ret.bitfld_ = (bitfld_ | rhs.bitfld_);
    return ret;
  }
  TaskID operator^ (const TaskID &rhs) const {
    TaskID ret;
    ret.bitfld_ = (bitfld_ ^ rhs.bitfld_);
    return ret;
  }
  TaskID operator& (const TaskID &rhs) const {
    TaskID ret;
    ret.bitfld_ = (bitfld_ & rhs.bitfld_);
    return ret;
  }

 private:
  std::bitset<NUMBER_TASKID_BITS> bitfld_;
};

//----------------------------------------------------------------------------------------
//! \class Task
//  \brief data and function object for an individual Task.
//  NOTE: the Task function must take arguments (Driver*, int stage)

class Task {
 public:
  Task(TaskID id, TaskID dep, std::function<TaskStatus(Driver*, int)> func) :
    myid_(id), dep_(dep), func_(func) {}
  TaskStatus operator()(Driver *d, int s) {return func_(d,s);}
  TaskID GetID() {return myid_;}
  TaskID GetDependency() {return dep_;}
  void SetComplete() {complete_ = true;}
  void SetIncomplete() {complete_ = false;}
  bool IsComplete() {return complete_;}
  // If this Task depends on id, change that dependency to 'newdep'
  void ChangeDependency(TaskID id, TaskID newdep) {
    if ((dep_ & id) == id) {dep_ = ((dep_ ^ id) | newdep);}
  }

 private:
  TaskID myid_;
  TaskID dep_;
  bool complete_ = false;
  std::function<TaskStatus(Driver*, int)> func_;
};

//----------------------------------------------------------------------------------------
//! \class TaskList
//  \brief data and function definitions for a task list

class TaskList {
 public:
  TaskList() = default;
  ~TaskList() = default;

  bool IsComplete() {
    for (auto &it : task_list_) {
      auto id = it.GetID();
      if (!(tasks_completed_.CheckDependencies(id))) return false;
    }
    return true;
  }
  int Size() {return task_list_.size();}
  bool Empty() {return task_list_.empty();}
  void MarkTaskComplete(TaskID id) { tasks_completed_.SetComplete(id); }
  TaskID GetIDLastTask() {return task_list_.back().GetID();}
  void PrintIDs() { for (auto &it : task_list_) {it.GetID().PrintID();} }
  void PrintDependencies() { for (auto &it : task_list_) {it.GetDependency().PrintID();} }

  void Reset() {
    tasks_completed_.Clear();
    for (auto &it : task_list_) { it.SetIncomplete(); }
  }

  // cycle through the task list once, do any tasks whose dependencies are clear
  TaskListStatus DoAvailable(Driver *d, int s) {
    for (auto &task : task_list_) {
      auto dep = task.GetDependency();
      if ( tasks_completed_.CheckDependencies(dep) && !(task.IsComplete()) ) {
        TaskStatus status = task(d,s);
        if (status == TaskStatus::complete) {
          task.SetComplete();
          MarkTaskComplete(task.GetID());
        }
      }
    }
    if (IsComplete()) return TaskListStatus::complete;
    return TaskListStatus::running;
  }

  // ADD new Task with ID, given dependency, and a pointer to a member function of
  // class T. Returns the ID of the new task. Task function must have arguments
  // (Driver*, int). Usage: taskid = tl.AddTask(&T::DoSomething, obj, dependency);
  template <class F, class T>
  TaskID AddTask(F func, T *obj, TaskID dep) {
    auto size = task_list_.size();
    TaskID id(size+1);
    task_list_.push_back( Task(id, dep,
       [=](Driver *d, int s) mutable -> TaskStatus {return (obj->*func)(d,s);}) );
    return id;
  }

  // ADD new Task with ID, given dependency, and a std::function.
  TaskID AddTask(std::function<TaskStatus(Driver*, int)> func, TaskID dep) {
    auto size = task_list_.size();
    TaskID id(size+1);
    task_list_.push_back(Task(id, dep, func));
    return id;
  }

 protected:
  std::list<Task> task_list_;
  TaskID tasks_completed_;
};

#endif  // TASKLIST_TASK_LIST_HPP_
