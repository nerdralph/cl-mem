# cl-mem

cl-mem is an OpenCL memory benchmark utility.

Version 0.2 tests sequential write and read speeds.
Version 0.3 added sequential copy.
Random read/write tests are planned for a later version.

example R9 380 with memory clocked at 1.5Ghz:

Running write test.
128 GB in 1150.1 ms (111.3 GB/s)
Running read test.
128 GB in 779.5 ms (164.2 GB/s)
Running copy test.
128 GB in 906.2 ms (141.3 GB/s)


# Thanks

cl-mem uses code from Marc Bevand's SILENTARMY zcash miner
