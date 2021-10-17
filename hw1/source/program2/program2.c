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
#include <linux/wait.h>

/* If WIFEXITED(STATUS), the low-order 8 bits of the status.  */
#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)
/* If WIFSIGNALED(STATUS), the terminating signal.  */
#define __WTERMSIG(status) ((status)&0x7f)
/* If WIFSTOPPED(STATUS), the signal that stopped the child.  */
#define __WSTOPSIG(status) __WEXITSTATUS(status)
/* Nonzero if STATUS indicates normal termination.  */
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)
/* Nonzero if STATUS indicates termination by a signal.  */
#define __WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)
/* Nonzero if STATUS indicates the child is stopped.  */
#define __WIFSTOPPED(status) (((status)&0xff) == 0x7f)

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
  const char path[] = "/home/nanfei/Documents/CSC3150/Assignment/CSC3150-Assgnt/hw1/source/program2/test";
  const char *const argv[] = {path, NULL, NULL};
  const char *const envp[] = {NULL};
  struct filename *my_filename = getname(path);
  printk("[program2] : Child process\n");
  result = do_execve(my_filename, argv, envp);
  if (!result)
    return 0;
  do_exit(result);
}

// implement wait function
void my_wait(pid_t pid)
{
  int status = 0;
  long result;
  struct wait_opts wo;
  struct pid *wo_pid = NULL;
  enum pid_type type;
  type = PIDTYPE_PID;
  wo_pid = find_get_pid(pid);

  wo.wo_type = type;
  wo.wo_pid = wo_pid;
  wo.wo_flags = WEXITED | WUNTRACED;
  wo.wo_info = NULL;
  wo.wo_stat = (int __user *)&status;
  wo.wo_rusage = NULL;

  result = do_wait(&wo);
  //printk("The result is %d\n", result);
  //printk("status address is %u\n", &status);
  //printk("status is %d\n", status);
  if (__WIFEXITED(status))
  {
    printk("[program2] : Child process normal termination with exit status = %d\n", __WEXITSTATUS(status));
    printk("[program2] : Child process terminated\n");
  }
  else if (__WIFSIGNALED(status))
  {
    switch (__WTERMSIG(status))
    {
    case 1:
    {
      printk("[program2] : child process get SIGHUP signal\n");
      break;
    }
    case 2:
    {
      printk("[program2] : child process get SIGINT signal\n");
      break;
    }
    case 3:
    {
      printk("[program2] : child process get SIGQUIT signal\n");
      break;
    }
    case 4:
    {
      printk("[program2] : child process get SIGILL signal\n");
      break;
    }
    case 5:
    {
      printk("[program2] : child process get SIGTRAP signal\n");
      break;
    }
    case 6:
    {
      printk("[program2] : child process get SIGABRT signal\n");
      break;
    }
    case 7:
    {
      printk("[program2] : child process get SIGBUS signal\n");
      break;
    }
    case 8:
    {
      printk("[program2] : child process get SIGFPE signal\n");
      break;
    }
    case 9:
    {
      printk("[program2] : child process get SIGKILL signal\n");
      break;
    }
    case 11:
    {
      printk("[program2] : child process get SIGSEGV signal\n");
      break;
    }
    case 13:
    {
      printk("[program2] : child process get SIGPIPE signal\n");
      break;
    }
    case 14:
    {
      printk("[program2] : child process get SIGALRM signal\n");
      break;
    }
    case 15:
    {
      printk("[program2] : child process get SIGTERM signal\n");
      break;
    }
    default:
      printk("[program2] : child process get %d signal\n", __WTERMSIG(status));
    }
    printk("[program2] : Child process terminated\n");
    printk("[program2] : The return signal is %d\n", __WTERMSIG(status));
  }
  else if (__WIFSTOPPED(status))
  {
    printk("[program2] : child process get SIGSTOP signal\n");
    printk("[program2] : Child process stopped\n");
    printk("[program2] : The return signal is %d\n", __WSTOPSIG(status));
  }

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
  pid = _do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0);

  /* execute a test program in child process */
  if (pid == 0)
  {
    my_exec();
  }
  printk("[program2] : The child process has pid = %d", pid);
  printk("[program2] : This is the parent process, pid = %d", (int)current->pid);
  /* wait until child process terminates */
  my_wait(pid);

  return 0;
}

static int __init program2_init(void)
{

  printk("[program2] : Module_init\n");
  struct task_struct *task;

  /* write your code here */

  /* create a kernel thread to run my_fork */
  printk("[program2] : Module_init create Kthread starts\n");
  task = kthread_create(&my_fork, NULL, "my thread");
  printk("[program2] : Module_init kthread start\n");
  if (!IS_ERR(task))
  {
    wake_up_process(task);
  }

  return 0;
}

static void __exit program2_exit(void) { printk("[program2] : Module_exit\n"); }

module_init(program2_init);
module_exit(program2_exit);
