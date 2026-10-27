# Execution contract

Include `fdaPDE/execution.h`, or use the `fdaPDE/core.h` aggregate.
The module depends only on the standard library and the shared assertion header.
It supplies asynchronous callables/futures, synchronous parallel loops and reductions,
and copyable task graphs. Set the worker count before the first runtime operation.
Call `parallel_join()` from outside executor workers to wait for all submitted work.

`parallel_async` transports callable exceptions through its future. `parallel_for`
waits for every submitted chunk before rethrowing a body or custom-step exception;
when several chunks fail, it reports the lowest failing iteration index. A failed
chunk may skip its remaining iterations. If callable construction or submission
fails partway through a loop, previously submitted chunks finish before that
submission failure is rethrown, taking precedence over any body failure.

Ordinary fire-and-forget callables, `parallel_for_each`, `parallel_reduce`, and graph
bodies must not throw: their exceptions escape worker execution and terminate the
process. Captured references must remain alive until completion. Shared
callbacks/reducers must be safe for concurrent invocation; graph topology must not
be mutated while it executes.
Reductions preserve chunk order and apply the seed once; the reducer must support
associative regrouping. They do not promise serial floating-point reproducibility.

Loop ranges, distances, chunk arithmetic and graph sizes must fit the implementation's
`int` counters. Custom steps must advance toward the end and terminate. The deque uses
64-bit indices and assumes those indices never overflow. Ranges are not widened
automatically, and deque indices are not rebased.

The process-lifetime singleton deliberately does not run its destructor. Local internal
executors must finish work before destruction; `stop()` does not complete pending tasks.
`parallel_for` drains partial submissions but does not roll back completed writes.
Other group APIs do not provide transactional recovery from partial submission,
and worker creation failure has no coordinated partial-startup recovery. Complete tasks are
accounted and destroyed before global join returns; graph-local waits cover graph
callable completion, and global join can be used before observing pool reclamation.

## Assertion classification

| Check | Primitive | Reason |
| --- | --- | --- |
| positive worker count and configuration freeze | strong | public configuration contract |
| join from an executor worker | strong | public call would deadlock |
| graph cycle, graph node index, cross-graph edge | strong | public graph validation |
| reserved task count and internal executor size | debug | internal preconditions |
| accounted completion and successful owner drain | debug | internal hot-path invariants |
| deque capacity shape | debug | internal buffer configuration |

Reservation precedes publication. Failed reservation destroys the allocated callable;
failed queue publication destroys it and rolls back accounting. Disabled debug checks
remove validation evaluation, while required publication, ownership and synchronization
operations remain active.

## Chase–Lev references

The read of the stolen slot must precede the successful CAS on `top`: otherwise the
owner can reuse that circular slot before the thief reads it. This ordering is already
required in Figure 2 and section 2.2 of [Chase and Lev (2005)](https://www.cs.wm.edu/~dcschmidt/PDF/work-stealing-dequeue.pdf).

The C11 presentation uses atomic buffer cells with relaxed slot loads/stores and a
slot read before CAS. Figure 1 publishes `bottom` with a release fence followed by a
relaxed store. The executor uses a release store to publish `bottom`; atomic slots and reading
before CAS address the separate risk of concurrent slot reuse.
See section 2 and Figure 1 of [Lê et al. (2013)](https://www.di.ens.fr/~zappa/readings/ppopp13.pdf).

Contention and wraparound tests exercise the bounded deque and atomic slots; they
do not prove linearizability under every C++ memory-model execution.
