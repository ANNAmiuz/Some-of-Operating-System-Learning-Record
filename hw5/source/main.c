#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include <linux/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0	 // Student ID
#define DMARWOKADDR 0x4		 // RW function complete
#define DMAIOCOKADDR 0x8	 // ioctl function complete
#define DMAIRQOKADDR 0xc	 // ISR function complete
#define DMACOUNTADDR 0x10	 // interrupt count function complete
#define DMAANSADDR 0x14		 // Computation answer
#define DMAREADABLEADDR 0x18 // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c	 // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20	 // data.a opcode
#define DMAOPERANDBADDR 0x21 // data.b operand1
#define DMAOPERANDCADDR 0x25 // data.c operand2

#define IRQ_NUM 1

//static int my_count = 0;

void *dma_buf;
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdevp;

int prime(int base, short nth)
{
	int fnd = 0;
	int i, num, isPrime;
	num = base;
	while (fnd != nth)
	{
		isPrime = 1;
		num++;
		for (i = 2; i <= num / 2; i++)
		{
			if (num % i == 0)
			{
				isPrime = 0;
				break;
			}
		}
		if (isPrime)
		{
			fnd++;
		}
	}
	return num;
}

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t *);
static int drv_open(struct inode *, struct file *);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t *);
static int drv_release(struct inode *, struct file *);
static long drv_ioctl(struct file *, unsigned int, unsigned long);

// cdev file_operations
static struct file_operations fops = {
	owner : THIS_MODULE,
	read : drv_read,
	write : drv_write,
	unlocked_ioctl : drv_ioctl,
	open : drv_open,
	release : drv_release,
};

// in and out function
void myoutc(unsigned char data, unsigned short int port);
void myouts(unsigned short data, unsigned short int port);
void myouti(unsigned int data, unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn
{
	char a;
	int b;
	short c;
} * dataIn;

// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct *ws);

// Input and output data from/to DMA
void myoutc(unsigned char data, unsigned short int port)
{
	*(volatile unsigned char *)(dma_buf + port) = data;
}
void myouts(unsigned short data, unsigned short int port)
{
	*(volatile unsigned short *)(dma_buf + port) = data;
}
void myouti(unsigned int data, unsigned short int port)
{
	*(volatile unsigned int *)(dma_buf + port) = data;
}
unsigned char myinc(unsigned short int port)
{
	return *(volatile unsigned char *)(dma_buf + port);
}
unsigned short myins(unsigned short int port)
{
	return *(volatile unsigned short *)(dma_buf + port);
}
unsigned int myini(unsigned short int port)
{
	return *(volatile unsigned int *)(dma_buf + port);
}

static int drv_open(struct inode *ii, struct file *ff)
{
	try_module_get(THIS_MODULE);
	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}
static int drv_release(struct inode *ii, struct file *ff)
{
	module_put(THIS_MODULE);
	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t *lo)
{
	/* Implement read operation for your device */
	unsigned int res = myini(DMAANSADDR);
	printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, res);
	put_user(res, (unsigned int *)buffer);
	myouti(0, DMAREADABLEADDR);
	myouti(0, DMAANSADDR);
	return 0;
}

static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t *lo)
{
	/* Implement write operation for your device */
	unsigned int isBlocking;

	/* MAKE A BUFFER AND WRITE DATA IN*/
	dataIn = kmalloc(sizeof(typeof(*dataIn)), GFP_KERNEL);
	copy_from_user(dataIn, buffer, ss);

	/* ASSIGN DATA TO PORTS */
	myoutc((unsigned char)dataIn->a, DMAOPCODEADDR);
	myouti((unsigned int)dataIn->b, DMAOPERANDBADDR);
	myouti((unsigned short)dataIn->c, DMAOPERANDCADDR);

	/* INIIALIZE WORK */
	INIT_WORK(work_routine, drv_arithmetic_routine);
	isBlocking = myini(DMABLOCKADDR);
	printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);

	// Decide io mode
	if (isBlocking == 0)
	{
		// printk("%s:%s(): queue work\n",PREFIX_TITLE, __func__);
		// printk("%s:%s(): non-block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
	}
	else if (isBlocking == 1)
	{
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine); // put the work task in global workqueue
		flush_scheduled_work();		 // flush work on work queue, force execution
	}

	return 0;
}


static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	/* Implement ioctl setting for your device */

	int cur_arg = 0, res;
	get_user(cur_arg, (int *)arg);

	switch (cmd)
	{
	case HW5_IOCSETSTUID:
		myouti(cur_arg, DMASTUIDADDR);
		res = myini(DMASTUIDADDR);
		printk("%s:%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, res);
		break;

	case HW5_IOCSETRWOK:
		myouti(cur_arg, DMARWOKADDR);
		res = myini(DMARWOKADDR);
		if (res == 1)
			printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
		break;

	case HW5_IOCSETIOCOK:
		myouti(cur_arg, DMAIOCOKADDR);
		res = myini(DMAIOCOKADDR);
		if (res == 1)
			printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
		break;

	case HW5_IOCSETIRQOK:
		myouti(cur_arg, DMAIRQOKADDR);
		res = myini(DMAIRQOKADDR);
		if (res == 1)
			printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
		break;

	case HW5_IOCSETBLOCK:
		myouti(cur_arg, DMABLOCKADDR);
		res = myini(DMABLOCKADDR);
		if (res == 1)
			printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
		else if (res == 0)
			printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
		break;

	case HW5_IOCWAITREADABLE:
		printk("%s:%s(): wait readable 1\n", PREFIX_TITLE, __func__);
		while (myini(DMAREADABLEADDR) == 0) msleep(314);
		res = myini(DMAREADABLEADDR);
		put_user(res, (int *)arg);
		break;

	default:
		break;
	}
	return 0;
}


// set and get back for debug
static void drv_arithmetic_routine(struct work_struct *ws)
{
	/* Implement arthemetic routine */
	unsigned int isBlocking;
	unsigned char opcode = myinc(DMAOPCODEADDR);
	
	/* OPERATION */
	unsigned int ans = 0;
	switch (opcode)
	{
	case '+':
		ans = myini(DMAOPERANDBADDR) + (unsigned int)myins(DMAOPERANDCADDR);
		myouti(ans, DMAANSADDR);
		break;
	case '-':
		ans = myini(DMAOPERANDBADDR) - (unsigned int)myins(DMAOPERANDCADDR);
		myouti(ans, DMAANSADDR);
		break;
	case '*':
		ans = myini(DMAOPERANDBADDR) * (unsigned int)myins(DMAOPERANDCADDR);
		myouti(ans, DMAANSADDR);
		break;
	case '/':
		ans = myini(DMAOPERANDBADDR) / (unsigned int)myins(DMAOPERANDCADDR);
		myouti(ans, DMAANSADDR);
		break;
	case 'p':
		ans = prime(myini(DMAOPERANDBADDR), myins(DMAOPERANDCADDR));
		myouti(ans, DMAANSADDR);
		break;
	default:
		ans = 0;
		myouti(ans, DMAANSADDR);
	}

	isBlocking = myini(DMABLOCKADDR);
	if (isBlocking == 0)
		myouti(1, DMAREADABLEADDR);
	printk("%s:%s(): %d %c %d = %d\n", PREFIX_TITLE, __func__, myini(DMAOPERANDBADDR), myinc(DMAOPCODEADDR), myins(DMAOPERANDCADDR), myini(DMAANSADDR));
	return;
}

/* BONUS: IRQ_HANDLER */
irqreturn_t handler(int irq_num, void *dev)
{
	// ++my_count;
	myouti(myini(DMACOUNTADDR) + 1, DMACOUNTADDR);
	return IRQ_HANDLED;
}

static int __init init_modules(void)
{
	dev_t dev;
	int ret = 0;
	typedef irqreturn_t (*irq_handler_t)(int, void *);

	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	/* Register chrdev */
	ret = alloc_chrdev_region(&dev, 0, 1, "mydev");
	if (ret)
	{
		printk("Cannot alloc chrdev\n");
		return ret;
	}

	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);
	printk("%s:%s(): register chrdev(%d,%d)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);

	/* Init cdev and make it alive */
	dev_cdevp = cdev_alloc();

	cdev_init(dev_cdevp, &fops);
	dev_cdevp->owner = THIS_MODULE;
	ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);
	if (ret < 0)
	{
		printk("Add chrdev failed\n");
		return ret;
	}

	/* BONUS: INIT IRQ, IRQ_NUM = 1 (PAGE 23) */
	printk("%s:%s(): request_irq %d returns %d\n", PREFIX_TITLE, __func__, IRQ_NUM, request_irq(IRQ_NUM, (irq_handler_t)handler, IRQF_SHARED, "OS_ASS5 DEVICE", (void *)dev_cdevp));

	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL); // simulate register and memory on device, kmalloc a dma buffer
	printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __func__);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	return 0;
}


static void __exit exit_modules(void)
{
	dev_t dev;
	/* BONUS : Free irq */
	free_irq(IRQ_NUM, (void *)dev_cdevp);
	printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, myini(DMACOUNTADDR));

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	dev = MKDEV(dev_major, dev_minor);
	cdev_del(dev_cdevp);
	unregister_chrdev_region(dev, 1);
	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);

	/* Free work routine */
	kfree(work_routine);

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
