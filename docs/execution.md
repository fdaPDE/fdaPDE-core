# Execution contract

Include `fdaPDE/execution.h`, or use the `fdaPDE/core.h` aggregate.
The module depends only on the standard library and the shared assertion header.
It supplies asynchronous callables/futures, synchronous parallel loops and reductions,
and copyable task graphs. Set the concurrency and optional affinity before the first
runtime operation. `parallel_set_num_threads(P)` configures total parallelism for a
single collaborating caller: for P >= 2 the runtime starts P - 1 background threads
and reserves one execution context for that caller. At P = 1 synchronous algorithms
execute serially on the caller; one background thread remains available for independent
asynchronous progress. Concurrent external callers and unrelated threads are not a
process-wide concurrency limit.

`parallel_initialize(max_threads)` starts the pool once, capped by its configured
capacity, and returns its actual parallelism. Later calls reuse that pool; a cap
smaller than the existing pool raises `std::logic_error`. Invalid nonpositive caps
raise `std::invalid_argument`. Use this operation for repeated model phases that
share the process-lifetime executor.

`parallel_set_affinity(true)` requests best-effort background CPU pinning on supported
operating systems. It defaults to false. Unsupported systems or OS binding failures
retain ordinary scheduling. Both configuration setters reject changes after runtime
initialization. Concurrency detection respects the calling thread's allowed CPU
mask and scheduler reservations, preferring `SLURM_CPUS_PER_TASK` over node-wide
counts. `OMP_NUM_THREADS` controls the separate OpenMP runtime and does not limit
this executor. Without affinity or scheduler information, detection uses the
hardware count with a minimum of one.

Call `parallel_join()` from outside executor workers to wait for all submitted work.
`parallel_async` transports callable exceptions through its future. `parallel_for`
waits for every submitted chunk before rethrowing a body or custom-step exception;
when several chunks fail, it reports the lowest failing iteration index. A failed
chunk may skip its remaining iterations. If callable construction or submission
fails partway through a loop, previously published chunks finish before that
submission failure is rethrown, taking precedence over any body failure. Prepared
chunks that have not yet been published are discarded.

`parallel_for_each`, `parallel_reduce`, and graph execution also drain published work
before propagating a callable exception. Graph successors still run after a node
fails; after the graph completes, one recorded node exception is rethrown. These APIs
do not select exceptions deterministically when several callbacks fail. Ordinary
fire-and-forget callables must not throw: their exceptions escape worker execution
and terminate the process. Captured references must remain alive until completion.
Shared callbacks/reducers must be safe for concurrent invocation; graph topology
must not be mutated while it executes.

Reductions preserve chunk order and apply the seed once; the reducer must support
associative regrouping. They do not promise serial floating-point reproducibility.
Loop ranges, distances, chunk arithmetic and graph sizes must fit the implementation's
`int` counters. Custom steps must advance toward the end and terminate. The deque uses
64-bit indices and assumes those indices never overflow. Ranges are not widened
automatically, and deque indices are not rebased.

The process-lifetime singleton deliberately does not run its destructor. Local internal
executors must finish work before destruction; `stop()` does not complete pending tasks.
Partial background-thread startup is cleaned up before a construction failure returns.
Group APIs drain partial publication but do not roll back completed writes. Completed
tasks release their captures and accounting before global join returns. Group-local
waits cover their callbacks; global join additionally covers pool reclamation.

## Assertion classification

| Check | Primitive | Reason |
| --- | --- | --- |
| positive parallelism and configuration freeze | strong | public configuration contract |
| join from an executor worker | strong | public call would deadlock |
| graph cycle, graph node index, cross-graph edge | strong | public graph validation |
| explicit affinity CPU specification | strong | public affinity validation |
| internal executor size, pool block size, deque capacity | debug | internal storage preconditions |

Task allocation reserves queue resources before publication. Failed preparation
reclaims the callable and unpublished mailbox node. Local publication uses reserved
deque capacity; mailbox publication requires no allocation. Accounting is owned by
execution contexts, with parked waiters notified after publication and completion.
Disabled debug checks remove validation evaluation, while required ownership and
synchronization operations remain active.

## Chase–Lev references

The read of the stolen slot must precede the successful CAS on `top`: otherwise the
owner can reuse that circular slot before the thief reads it. This ordering is required
in Figure 2 and section 2.2 of [Chase and Lev (2005)](https://www.cs.wm.edu/~dcschmidt/PDF/work-stealing-dequeue.pdf).
The C11 presentation uses atomic buffer cells with relaxed slot loads/stores and a
slot read before CAS. Figure 1 publishes `bottom` with a release fence followed by a
relaxed store. The executor uses a release store to publish `bottom`; atomic slots and
reading before CAS address the separate risk of concurrent slot reuse.
See section 2 and Figure 1 of [Lê et al. (2013)](https://www.di.ens.fr/~zappa/readings/ppopp13.pdf).
Contention and wraparound tests exercise the bounded deque and atomic slots; they
do not prove linearizability under every C++ memory-model execution.
