#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

struct wait_opts
{
  enum pid_type wo_type;
  int wo_flags;
  struct pid *wo_pid;

  struct siginfo __user *wo_info;
  int __user *wo_stat;
  struct rusage __user *wo_rusage;

  wait_queue_t child_wait;
  int notask_error;
};

extern long do_wait(struct wait_opts *wo);
extern long _do_fork(unsigned long clone_flags,
                     unsigned long stack_start,
                     unsigned long stack_size,
                     int __user *parent_tidptr,
                     int __user *child_tidptr,
                     unsigned long tls);
extern void do_exit(long code);
extern int do_execve(struct filename *filename,
                     const char __user *const __user *__argv,
                     const char __user *const __user *__envp);
extern struct filename *getname(const char __user *filename);

// implement execve function
int my_exec(void)
{
  int result;
  const char path[] = "/home/nanfei/Documents/CSC3150/assignment/hw1/source/program2/test";
  const char *const argv[] = {path, NULL, NULL};
  const char *const envp[] = {NULL};
  struct filename *my_filename = getname(path);
  result = do_execve(my_filename, argv, envp);
  if (!result)
    return 0;
  do_exit(result);
}

// implement wait function
void my_wait(pid_t pid)
{
  int status;
  long result;
  struct wait_opts wo;
  struct pid *wo_pid = NULL;
  enum pid_type type;
  type = PIDTYPE_PID;
  wo_pid = find_get_pid(pid);

  wo.wo_type = type;
  wo.wo_pid = wo_pid;
  wo.wo_flags = WEXITED;
  wo.wo_info = NULL;
  wo.wo_stat = (int __user *)&status;
  wo.wo_rusage = NULL;

  result = do_wait(&wo);
  printk("do_wait return valus is %d\n", &result);
  // output child process exit status
  printk("[Do_Fork] : The return signal is %d\n", *wo.wo_stat);
  put_pid(wo_pid);
  return;
}

// implement exit function
void my_exit(long code)
{
  do_exit(code);
}

// implement fork function
int my_fork(void *argc)
{

  // set default sigaction for current process
  int i, pid;
  struct k_sigaction *k_action = &current->sighand->action[0];
  for (i = 0; i < _NSIG; i++)
  {
    k_action->sa.sa_handler = SIG_DFL;
    k_action->sa.sa_flags = 0;
    k_action->sa.sa_restorer = NULL;
    sigemptyset(&k_action->sa.sa_mask);
    k_action++;
  }

  /* fork a process using do_fork */
  pid = do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0);

  /* execute a test program in child process */
  if (pid == 0)
  {
    my_exec();
  }

  /* wait until child process terminates */
  my_wait(pid);

  return 0;
}

static int __init program2_init(void)
{

  printk("[program2] : Module_init\n");
  struct task_struct task;

  /* write your code here */

  /* create a kernel thread to run my_fork */
  task = kthread_create(&my_fork, NULL, "my thread");
  if (!IS_ERR(task))
  {
    printk("Kthread starts\n");
    wake_up_process(task);
  }

  return 0;
}

static void __exit program2_exit(void) { printk("[program2] : Module_exit\n"); }

module_init(program2_init);
module_exit(program2_exit);
