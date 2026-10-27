# Verifiche delle API preallocate native

Baseline develop-SIMD: e14d9e41b45186b5cd7984712be504c1799fd6bb.
Le modifiche restano locali; nessuna modifica RGCCA o pubblicazione.

- CTest integrato GCC15 SIMD ON: 132/132, exit 0. Ricompilati serialmente i target dense/sparse e i rispettivi header check; gli altri target non toccati sono stati eseguiti dalle build gia presenti. Log build, CTest e JUnit conservati.
- Test mirati GCC15: dense 7/7 OFF e ON, sparse 8/8 OFF e ON. I dettagli riproducibili dense sono in focused-dense/. La suite integrata e i sanitizer includono i test sparse e i 3 test replay.
- Un controllo mirato FDAPDE_NO_DEBUG passa per la forma incompatibile prima di scrivere. Le warning preesistenti dei parametri/locali usati solo nelle assert sono rimaste visibili nella build di questo singolo controllo, come spiegato in focused-dense/README.md.
- Apple Clang21 ASan/UBSan SIMD ON: dense 7/7, sparse/replay 11/11; nessun diagnostic runtime. Comandi, flags, log/XML e hash in sanitizers/. Core e test sono instrumentati; le librerie GTest gia disponibili non sono state ricompilate con sanitizer.
- Benchmark GCC15/Eigen3.4: 143 casi, 575 confronti tutti verified; log in campaign.log, dati/manifest nel run sibling native-preallocated-20261002/. Non sono misure dell'intera applicazione.

Build e benchmark sono stati serializzati in background/nice19. Nessun fast-math, nuovo flag ISA o pacchetto installato. final-artifact-check.json ricontrolla sorgenti/input/header Eigen/binari dopo la generazione del report. I comandi dei test usano i percorsi dei compiler/GTest gia presenti su questo host; per la campagna portabile usare tests/benchmarks/README_PREALLOCATED.md.
