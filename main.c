#define _GNU_SOURCE	1/* memrchr */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>
#include <CL/cl.h>
#include "_kernel.h"
#include "config.h"

typedef uint8_t		uchar;
typedef uint32_t	uint;

const char* test_names[] = {"write", "read", "copy"};

int         verbose = 0;
uint32_t	do_list_devices = 0;
uint32_t	gpu_to_use = 0;

typedef struct  debug_s
{
    uint32_t    dropped_coll;
    uint32_t    dropped_stor;
}               debug_t;

void debug(const char *fmt, ...)
{
    va_list     ap;
    if (!verbose)
        return ;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

void warn(const char *fmt, ...)
{
    va_list     ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// fatal error
void fatal(const char *fmt, ...)
{
    va_list     ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    exit(1);
}

#define CL_CHECK(STATUS)    if (STATUS != CL_SUCCESS) fatal("Error (%d) at line%d\n", STATUS, __LINE__)
 
uint64_t parse_num(char *str)
{
    char	*endptr;
    uint64_t	n;
    n = strtoul(str, &endptr, 0);
    if (endptr == str || *endptr)
	fatal("'%s' is not a valid number\n", str);
    return n;
}

uint64_t now(void)
{
    struct timeval	tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}

void show_time(uint64_t t0)
{
    uint64_t            t1;
    t1 = now();
    fprintf(stderr, "Elapsed time: %.1f msec\n", (t1 - t0) / 1e3);
}

void set_blocking_mode(int fd, int block)
{
    int		f;
    if (-1 == (f = fcntl(fd, F_GETFL)))
	fatal("fcntl F_GETFL: %s\n", strerror(errno));
    if (-1 == fcntl(fd, F_SETFL, block ? (f & ~O_NONBLOCK) : (f | O_NONBLOCK)))
	fatal("fcntl F_SETFL: %s\n", strerror(errno));
}

void randomize(void *p, ssize_t l)
{
    const char	*fname = "/dev/urandom";
    int		fd;
    ssize_t	ret;
    if (-1 == (fd = open(fname, O_RDONLY)))
	fatal("open %s: %s\n", fname, strerror(errno));
    if (-1 == (ret = read(fd, p, l)))
	fatal("read %s: %s\n", fname, strerror(errno));
    if (ret != l)
	fatal("%s: short read %d bytes out of %d\n", fname, ret, l);
    if (-1 == close(fd))
	fatal("close %s: %s\n", fname, strerror(errno));
}

cl_mem check_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
	void *host_ptr)
{
    cl_int	status;
    cl_mem	ret;
    ret = clCreateBuffer(ctx, flags, size, host_ptr, &status);
    if (status != CL_SUCCESS || !ret)
	    fatal("clCreateBuffer (%d)\n", status);
    return ret;
}

void check_clSetKernelArg(cl_kernel k, cl_uint a_pos, cl_mem *a)
{
    cl_int	status;
    status = clSetKernelArg(k, a_pos, sizeof (*a), a);
    if (status != CL_SUCCESS)
	    fatal("clSetKernelArg (%d)\n", status);
}

void check_clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel k, cl_uint
	work_dim, const size_t *global_work_offset, const size_t
	*global_work_size, const size_t *local_work_size, cl_uint
	event_wait_list_size, const cl_event *event_wait_list, cl_event
	*event)
{
    cl_int	status;
    status = clEnqueueNDRangeKernel(queue, k, work_dim, global_work_offset,
	    global_work_size, local_work_size, event_wait_list_size,
	    event_wait_list, event);
    if (status != CL_SUCCESS)
	    fatal("clEnqueueNDRangeKernel (%d)\n", status);
}

void check_clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, cl_bool
	blocking_read, size_t offset, size_t size, void *ptr, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
    cl_int	status;
    status = clEnqueueReadBuffer(queue, buffer, blocking_read, offset,
	    size, ptr, num_events_in_wait_list, event_wait_list, event);
    if (status != CL_SUCCESS)
        fatal("clEnqueueReadBuffer (%d)\n", status);
}

void hexdump(uint8_t *a, uint32_t a_len)
{
    for (uint32_t i = 0; i < a_len; i++)
        fprintf(stderr, "%02x", a[i]);
}

char *s_hexdump(const void *_a, uint32_t a_len)
{
    const uint8_t	*a = _a;
    static char		buf[4096];
    uint32_t		i;
    for (i = 0; i < a_len && i + 2 < sizeof (buf); i++)
        sprintf(buf + i * 2, "%02x", a[i]);
    buf[i * 2] = 0;
    return buf;
}

uint8_t hex2val(const char *base, size_t off)
{
    const char          c = base[off];
    if (c >= '0' && c <= '9')           return c - '0';
    else if (c >= 'a' && c <= 'f')      return 10 + c - 'a';
    else if (c >= 'A' && c <= 'F')      return 10 + c - 'A';
    fatal("Invalid hex char at offset %zd: ...%c...\n", off, c);
    return 0;
}

void get_program_build_log(cl_program program, cl_device_id device)
{
    cl_int		status;
    char	        val[2*1024*1024];
    size_t		ret = 0;
    status = clGetProgramBuildInfo(program, device,
	    CL_PROGRAM_BUILD_LOG,
	    sizeof (val),	// size_t param_value_size
	    &val,		// void *param_value
	    &ret);		// size_t *param_value_size_ret
    if (status != CL_SUCCESS)
	fatal("clGetProgramBuildInfo (%d)\n", status);
    fprintf(stderr, "%s\n", val);
}

void dump(const char *fname, void *data, size_t len)
{
    int			fd;
    ssize_t		ret;
    if (-1 == (fd = open(fname, O_WRONLY | O_CREAT | O_TRUNC, 0666)))
	fatal("%s: %s\n", fname, strerror(errno));
    ret = write(fd, data, len);
    if (ret == -1)
	fatal("write: %s: %s\n", fname, strerror(errno));
    if ((size_t)ret != len)
	fatal("%s: partial write\n", fname);
    if (-1 == close(fd))
	fatal("close: %s: %s\n", fname, strerror(errno));
}

void get_program_bins(cl_program program)
{
    cl_int		status;
    size_t		sizes;
    unsigned char	*p;
    size_t		ret = 0;
    status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
	    sizeof (sizes),	// size_t param_value_size
	    &sizes,		// void *param_value
	    &ret);		// size_t *param_value_size_ret
    if (status != CL_SUCCESS)
	fatal("clGetProgramInfo(sizes) (%d)\n", status);
    if (ret != sizeof (sizes))
	fatal("clGetProgramInfo(sizes) did not fill sizes (%d)\n", status);
    debug("Program binary size is %zd bytes\n", sizes);
    p = (unsigned char *)malloc(sizes);
    status = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
	    sizeof (p),	// size_t param_value_size
	    &p,		// void *param_value
	    &ret);	// size_t *param_value_size_ret
    if (status != CL_SUCCESS)
	fatal("clGetProgramInfo (%d)\n", status);
    dump("dump.co", p, sizes);
    debug("program: %02x%02x%02x%02x...\n", p[0], p[1], p[2], p[3]);
}

void print_platform_info(cl_platform_id plat)
{
    char	name[1024];
    size_t	len = 0;
    int		status;
    status = clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof (name), &name,
	    &len);
    if (status != CL_SUCCESS)
	fatal("clGetPlatformInfo (%d)\n", status);
    printf("Devices on platform \"%s\":\n", name);
    fflush(stdout);
}

void print_device_info(unsigned i, cl_device_id d)
{
    char	name[1024];
    size_t	len = 0;
    int		status;
    status = clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof (name), &name, &len);
    if (status != CL_SUCCESS)
	fatal("clGetDeviceInfo (%d)\n", status);
    printf("  ID %d: %s\n", i, name);
    fflush(stdout);
}

// non-debug version
void examine_ht(unsigned round, cl_command_queue queue, cl_mem buf_ht)
{
    (void)round;
    (void)queue;
    (void)buf_ht;
}

void examine_dbg(cl_command_queue queue, cl_mem buf_dbg, size_t dbg_size)
{
    debug_t     *dbg;
    size_t      dropped_coll_total, dropped_stor_total;
    if (verbose < 2)
	return ;
    dbg = (debug_t *)malloc(dbg_size);
    if (!dbg)
	fatal("malloc: %s\n", strerror(errno));
    check_clEnqueueReadBuffer(queue, buf_dbg,
            CL_TRUE,	// cl_bool	blocking_read
            0,		// size_t	offset
            dbg_size,   // size_t	size
            dbg,	// void		*ptr
            0,		// cl_uint	num_events_in_wait_list
            NULL,	// cl_event	*event_wait_list
            NULL);	// cl_event	*event
    dropped_coll_total = dropped_stor_total = 0;
    for (unsigned tid = 0; tid < dbg_size / sizeof (*dbg); tid++)
      {
        dropped_coll_total += dbg[tid].dropped_coll;
        dropped_stor_total += dbg[tid].dropped_stor;
	if (0 && (dbg[tid].dropped_coll || dbg[tid].dropped_stor))
	    debug("thread %6d: dropped_coll %zd dropped_stor %zd\n", tid,
		    dbg[tid].dropped_coll, dbg[tid].dropped_stor);
      }
    debug("Dropped: %zd (coll) %zd (stor)\n",
            dropped_coll_total, dropped_stor_total);
    free(dbg);
}

/*
** Sort a pair of binary blobs (a, b) which are consecutive in memory and
** occupy a total of 2*len 32-bit words.
**
** a            points to the pair
** len          number of 32-bit words in each pair
*/
void sort_pair(uint32_t *a, uint32_t len)
{
    uint32_t    *b = a + len;
    uint32_t     tmp, need_sorting = 0;
    for (uint32_t i = 0; i < len; i++)
	if (need_sorting || a[i] > b[i])
	  {
	    need_sorting = 1;
	    tmp = a[i];
	    a[i] = b[i];
	    b[i] = tmp;
	  }
	else if (a[i] < b[i])
	    return ;
}

/*
** Read a complete line from stdin. If 2 or more lines are available, store
** only the last one in the buffer.
**
** buf		buffer to store the line
** len		length of the buffer
** block	blocking mode: do not return until a line was read
**
** Return 1 iff a line was read.
*/
int read_last_line(char *buf, size_t len, int block)
{
    char	*start;
    size_t	pos = 0;
    ssize_t	n;
    set_blocking_mode(0, block);
    while (42)
      {
	n = read(0, buf + pos, len - pos);
	if (n == -1 && errno == EINTR)
	    continue ;
	else if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK))
	  {
	    if (!pos)
		return 0;
	    warn("strange: a partial line was read\n");
	    // a partial line was read, continue reading it in blocking mode
	    // to be sure to read it completely
	    set_blocking_mode(0, 1);
	    continue ;
	  }
	else if (n == -1)
	    fatal("read stdin: %s\n", strerror(errno));
	else if (!n)
	    fatal("EOF on stdin\n");
	pos += n;
	if (buf[pos - 1] == '\n')
	    // 1 (or more) complete lines were read
	    break ;
      }
    start = memrchr(buf, '\n', pos - 1);
    if (start)
      {
	warn("strange: more than 1 line was read\n");
	// more than 1 line; copy the last line to the beginning of the buffer
	pos -= (start + 1 - buf);
	memmove(buf, start + 1, pos);
      }
    // overwrite '\n' with NUL
    buf[pos - 1] = 0;
    return 1;
}

void run_opencl(cl_context ctx, cl_command_queue queue, cl_kernel* tests)
{
    cl_mem              buf_test, buf_dbg;
    void                *dbg = NULL;
    size_t  global_ws = 256*WAVES;
    size_t  local_ws = 64;
#ifdef ENABLE_DEBUG
    size_t              dbg_size = NR_ROWS * sizeof (debug_t);
#else
    size_t              dbg_size = 1 * sizeof (debug_t);
#endif
    //uint64_t		total;
    if (verbose)
	    fprintf(stderr, "Test buffers will use GB\n");
	//fprintf(stderr, "Hash tables will use %.1f MB\n", 2.0 * HT_SIZE / 1e6);
    // Set up buffers for the host and memory objects for the kernel
    if (!(dbg = calloc(dbg_size, 1)))
	    fatal("malloc: %s\n", strerror(errno));
    buf_dbg = check_clCreateBuffer(ctx, CL_MEM_READ_WRITE |
	    CL_MEM_COPY_HOST_PTR, dbg_size, dbg);
    buf_test = check_clCreateBuffer(ctx, CL_MEM_READ_WRITE, MEMSIZE, NULL);

    uint num_tests = sizeof(test_names)/sizeof(test_names[0]);
    for (unsigned test = 0; test < num_tests; test++)
    {
        check_clSetKernelArg(tests[test], 0, &buf_test);
        fprintf(stderr, "Running %s test.\n", test_names[test]);
        uint64_t t0 = now();
        check_clEnqueueNDRangeKernel(queue, tests[test], 1, NULL,
	        &global_ws, &local_ws, 0, NULL, NULL);
        cl_int status = clFinish(queue);
        CL_CHECK(status);
        uint64_t t1 = now();
        fprintf(stderr, "%d GB in %.1f ms (%.1f GB/s)\n", REPS,
	            (t1 - t0) / 1e3, REPS / ((t1 - t0) / 1e6));
    }
    // Clean up
    if (dbg)
        free(dbg);
    clReleaseMemObject(buf_dbg);
    clReleaseMemObject(buf_test);
}

/*
** Scan the devices available on this platform. Try to find the device
** selected by the "--use <id>" option and, if found, store the platform and
** device in plat_id and dev_id.
**
** plat			platform being scanned
** nr_devs_total	total number of devices detected so far, will be
** 			incremented by the number of devices available on this
** 			platform
** plat_id		where to store the platform id
** dev_id		where to store the device id
**
** Return 1 iff the selected device was found.
*/
unsigned scan_platform(cl_platform_id plat, cl_uint *nr_devs_total,
	cl_platform_id *plat_id, cl_device_id *dev_id)
{
    cl_device_type	typ = CL_DEVICE_TYPE_ALL;
    cl_uint		nr_devs = 0;
    cl_device_id	*devices;
    cl_int		status;
    unsigned		found = 0;
    unsigned		i;
    if (do_list_devices)
	print_platform_info(plat);
    status = clGetDeviceIDs(plat, typ, 0, NULL, &nr_devs);
    if (status != CL_SUCCESS)
	fatal("clGetDeviceIDs (%d)\n", status);
    if (nr_devs == 0)
	return 0;
    devices = (cl_device_id *)malloc(nr_devs * sizeof (*devices));
    status = clGetDeviceIDs(plat, typ, nr_devs, devices, NULL);
    if (status != CL_SUCCESS)
	fatal("clGetDeviceIDs (%d)\n", status);
    i = 0;
    while (i < nr_devs)
      {
	if (do_list_devices)
	    print_device_info(*nr_devs_total, devices[i]);
	else if (*nr_devs_total == gpu_to_use)
	  {
	    found = 1;
	    *plat_id = plat;
	    *dev_id = devices[i];
	    break ;
	  }
	(*nr_devs_total)++;
	i++;
      }
    free(devices);
    return found;
}

/*
** Stores the platform id and device id that was selected by the "--use <id>"
** option.
**
** plat_id		where to store the platform id
** dev_id		where to store the device id
*/
void scan_platforms(cl_platform_id *plat_id, cl_device_id *dev_id)
{
    cl_uint		nr_platforms;
    cl_platform_id	*platforms;
    cl_uint		i, nr_devs_total;
    cl_int		status;
    status = clGetPlatformIDs(0, NULL, &nr_platforms);
    if (status != CL_SUCCESS)
	fatal("Cannot get OpenCL platforms (%d)\n", status);
    if (!nr_platforms || verbose)
	fprintf(stderr, "Found %d OpenCL platform(s)\n", nr_platforms);
    if (!nr_platforms)
	exit(1);
    platforms = (cl_platform_id *)malloc(nr_platforms * sizeof (*platforms));
    if (!platforms)
	fatal("malloc: %s\n", strerror(errno));
    status = clGetPlatformIDs(nr_platforms, platforms, NULL);
    if (status != CL_SUCCESS)
	fatal("clGetPlatformIDs (%d)\n", status);
    i = nr_devs_total = 0;
    while (i < nr_platforms)
      {
	if (scan_platform(platforms[i], &nr_devs_total, plat_id, dev_id))
	    break ;
	i++;
      }
    if (do_list_devices)
	    exit(0);
    debug("Using GPU device ID %d\n", gpu_to_use);
    free(platforms);
}

void run_bench()
{
    cl_platform_id	plat_id = 0;
    cl_device_id    dev_id = 0;
//    cl_kernel		k_rounds[PARAM_K];
    uint num_tests = sizeof(test_names)/sizeof(test_names[0]);
    cl_kernel   tests[num_tests];
    cl_int		status;
    scan_platforms(&plat_id, &dev_id);
    if (!plat_id || !dev_id)
	fatal("Selected device (ID %d) not found; see --list\n", gpu_to_use);
    /* Create context.*/
    cl_context context = clCreateContext(NULL, 1, &dev_id,
	    NULL, NULL, &status);
    if (status != CL_SUCCESS || !context)
	fatal("clCreateContext (%d)\n", status);
    /* Creating command queue associate with the context.*/
    cl_command_queue queue = clCreateCommandQueue(context, dev_id,
	    0, &status);
    if (status != CL_SUCCESS || !queue)
	fatal("clCreateCommandQueue (%d)\n", status);
    /* Create program object */
    cl_program program;
    const char *source;
    size_t source_len;
    source = ocl_code;
    source_len = strlen(ocl_code);
    program = clCreateProgramWithSource(context, 1, (const char **)&source,
	    &source_len, &status);
    if (status != CL_SUCCESS || !program)
	fatal("clCreateProgramWithSource (%d)\n", status);
    /* Build program. */
    if (verbose)
	    fprintf(stderr, "Building program\n");
    status = clBuildProgram(program, 1, &dev_id,
	    "", // compile options
	    NULL, NULL);
    if (status != CL_SUCCESS)
    {
        warn("OpenCL build failed (%d). Build log follows:\n", status);
        get_program_build_log(program, dev_id);
	    exit(1);
    }
    //get_program_bins(program);
    // Create kernel objects
    //cl_kernel k_init_ht = clCreateKernel(program, "kernel_init_ht", &status);
    //if (status != CL_SUCCESS || !k_init_ht)
	//    fatal("clCreateKernel (%d)\n", status);
    for (unsigned test = 0; test < num_tests; test++)
    {
	    char	name[128];
	    // snprintf(name, sizeof (name), "kernel_round%d", round);
	    snprintf(name, sizeof (name), "test%d", test);
	    tests[test] = clCreateKernel(program, name, &status);
	    if (status != CL_SUCCESS || !tests[test])
	        fatal("clCreateKernel (%d)\n", status);
    }
    // Run
    run_opencl(context, queue, tests);
    // Release resources
    assert(CL_SUCCESS == 0);
    status = CL_SUCCESS;
    for (unsigned test = 0; test < num_tests; test++)
	    status |= clReleaseKernel(tests[test]);
    status |= clReleaseProgram(program);
    status |= clReleaseCommandQueue(queue);
    status |= clReleaseContext(context);
    if (status)
	    fprintf(stderr, "Cleaning resources failed\n");
}

enum
{
    OPT_HELP,
    OPT_VERBOSE,
    OPT_NONCES,
    OPT_THREADS,
    OPT_LIST,
    OPT_USE,
};

static struct option    optlong[] =
{
      {"help",		no_argument,		0,	OPT_HELP},
      {"h",		no_argument,		0,	OPT_HELP},
      {"verbose",	no_argument,		0,	OPT_VERBOSE},
      {"v",		no_argument,		0,	OPT_VERBOSE},
      {"t",		required_argument,	0,	OPT_THREADS},
      {"list",		no_argument,		0,	OPT_LIST},
      {"use",		required_argument,	0,	OPT_USE},
      {0,		0,			0,	0},
};

void usage(const char *progname)
{
    printf("Usage: %s [options]\n"
	    "OpenCL memory benchmark v0.3\n"
	    "\n"
	    "Options are:\n"
            "  -h, --help     display this help and exit\n"
            "  -v, --verbose  print verbose messages\n"
            "  --list         list available OpenCL devices by ID (GPUs...)\n"
            "  --use <id>     use GPU <id> (default: 0)\n"
            , progname);
}

int main(int argc, char **argv)
{
    int32_t             i;
    while (-1 != (i = getopt_long_only(argc, argv, "", optlong, 0)))
        switch (i)
        {
            case OPT_HELP:
                usage(argv[0]), exit(0);
                break ;
            case OPT_VERBOSE:
                verbose += 1;
                break ;
            case OPT_THREADS:
                // ignored, this is just to conform to the contest CLI API
                break ;
	        case OPT_LIST:
		        do_list_devices = 1;
		        break ;
	        case OPT_USE:
		        gpu_to_use = parse_num(optarg);
		        break ;
            default:
                fatal("Try '%s --help'\n", argv[0]);
                break ;
        }
    run_bench();
    return 0;
}
