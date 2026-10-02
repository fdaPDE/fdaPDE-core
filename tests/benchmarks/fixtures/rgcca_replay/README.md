# Snapshot RGCCA per replay offline

Quattro capsule accettate della componente 1, lambda 1 o 1000, dal run
`100206-performance-20261001`: 426/581/666/1418 righe e
7362/10043/11584/25242 coefficienti non nulli. I 20 file occupano 1.830.464 byte.

Ogni prefisso contiene Omega fisica (`-omega.mtx`), segnale `-c.bin`, direzione
iniziale `-warm.bin`, peso normalizzato accettato `-weight.bin` e marcatore
`.ready`. I file conservano esattamente i byte delle copie offline originali;
`manifest.json` registra SHA256, dimensioni, metadati e provenienza relativa.
Il marcatore e conservato come provenienza e non modifica il caricamento numerico.

Omega usa MatrixMarket coordinate real general, indici a base uno e entrambe
le triangolari. I vettori sono IEEE binary64 little endian senza intestazione,
con esattamente 8 byte per riga. Scaling e inizializzazione warm sono descritti
nel manifest e applicati fuori dalla misura dei kernel. I buffer nativi ed
Eigen sono preparati separatamente dagli stessi dati canonici.

Queste capsule non includono X, un bootstrap completo, casi della componente 2
lambda 10 o candidati rifiutati. Il replay FISTA non riproduce i gate LDLT/diretti,
le fattorizzazioni dei supporti o il pruning della produzione; non certifica
le decisioni applicative su lambda/pruning. I conteggi storici di iterazioni e
restart sono metadati, non risultati attesi del replay.

Il launcher Kami verifica byte e hash prima di qsub e sul worker e usa questo
manifest per default. Per i comandi e i limiti del confronto vedere
[README_PREALLOCATED.md](../../README_PREALLOCATED.md).
