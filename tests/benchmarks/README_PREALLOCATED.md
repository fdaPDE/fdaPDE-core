# Prodotti preallocati e confronto con Eigen

Il core espone tre percorsi aggiuntivi, senza dipendenze esterne:

```cpp
#include <fdaPDE/sparse_linear_algebra.h>

// input e output sono owner o viste native, gia dimensionati
omega.multiply_into(input, output);
X.multiply_into(weights, output);
X.transpose().multiply_into(values, output_transpose);

// tutte le letture dell'espressione devono provenire da buffer disgiunti dall'output
next.assign_disjoint((y - gradient / L).cwise().apply(
    [](double value) { return std::max(0.0, value); }));
```

`multiply_into` verifica permanentemente le dimensioni e non ridimensiona l'output. Il caso disgiunto con operandi plain non alloca; la sovrapposizione con input o matrice usa un risultato temporaneo. Le matrici dense non plain conservano il fallback con snapshot. La trasposta plain legge gli stride della matrice originale: la trasposta era gia lazy, il risparmio aggiunto riguarda il risultato e il percorso di accesso.

`assign_disjoint` verifica permanentemente la forma esatta e riusa l'executor di assegnazione. Non cerca alias nelle espressioni: la disgiunzione e un contratto del chiamante. Le assegnazioni ordinarie conservano lo snapshot sicuro. Queste API esplicite sono disponibili anche con SIMD OFF; `FDAPDE_ENABLE_SIMD=1`, oppure `FDAPDE_ENABLE_SIMD_PRODUCT=1` e `FDAPDE_ENABLE_SIMD_ASSIGNMENT=1`, selezionano i loop gia previsti dal core. I flag sono OFF per default e non disattivano l'autovettorizzazione del compilatore o di Eigen. SpMV preallocata conserva il kernel CSR ordinato: il suo beneficio atteso e il riuso del buffer, non una nuova accelerazione degli accessi indiretti.

## Esecuzione riproducibile

Sono richiesti C++20, Python 3.9+ ed Eigen **3.4.0 gia installato**, utilizzato soltanto dal benchmark. Non viene installato alcun pacchetto. Il runner compila serialmente OFF/ON con lo stesso compilatore e gli stessi flag, imposta i thread interni a uno e alterna le coppie AB/BA. Conserva manifest, SHA256 di sorgenti/input/binari, comandi, log, osservazioni JSONL, CSV e Markdown; anche un fallimento conserva lo stato parziale.

```bash
python3 tests/benchmarks/run_preallocated_comparison.py --self-test
python3 tests/benchmarks/run_preallocated_comparison.py \
  --compiler g++ --eigen-include "$PATH_EIGEN_INCLUDE" \
  --output output/simd/eigen-comparison/run \
  --pairs 5 --rounds 5 --round-ms 10 --timeout 900
```

Per un controllo locale breve su macOS aggiungere `--quick --background`. La policy background e `nice 19` riduce la priorita; non garantisce un core esclusivo. `--ops` e `--max-size` consentono un sottoinsieme; `--cpu` su Linux accetta soltanto un CPU presente nell'affinita consentita.

Il programma produce rapporti **prima/dopo**: un valore maggiore di uno indica che l'implementazione dopo i due punti e piu veloce. Pertanto `off/native:on/native` misura il guadagno del flag; `on/native:off/eigen-col` maggiore di uno indica un vantaggio di Eigen. Min/max sono dispersione delle coppie osservate, non intervalli di confidenza. API pubbliche e preallocate sono righe separate. Il probe conta `operator new` nativo fuori timing; Eigen usa anche `malloc`, quindi il suo numero e riportato come non disponibile.

Le forme sintetiche sono dispari, con offset di un double e densita sparse diverse. Lo sweep completo include dense fino a 4097x4095 (circa 128 MiB di soli coefficienti), CSR fino a 524289 righe e 65 nnz/riga (circa 392 MiB), e vettori fino a 1048577 coefficienti. Questi casi superano le cache tipiche, ma non costituiscono automaticamente un plateau: estendere lo sweep se il rapporto continua a cambiare sull'hardware misurato. Il working set riportato riguarda il kernel del backend attivo; per il replay e una stima del loop FISTA e non include i triplet canonici/raw e i dati di riferimento letti dal certificato finale. I processi conservano anche le rappresentazioni preparate per gli altri backend e gli oracle.

## Capsule reali e replay

Le copie offline in `output/simd/replay-inputs/manifest.json` sono quattro side della **componente 1**, lambda 1 o 1000, con 426/581/666/1418 righe e 7362/10043/11584/25242 nnz. Non rappresentano la componente 2 a lambda 10. Sorgenti e copie hanno SHA256 uguali; non si riesegue il bootstrap. Aggiungere uno o piu prefissi:

```bash
--input output/simd/replay-inputs/lambda1_initial/side-3 \
--input output/simd/replay-inputs/lambda1_restart/side-50 \
--input output/simd/replay-inputs/lambda1000_slow/side-0 \
--input output/simd/replay-inputs/lambda1_large/side-18
```

I dati canonici sono scalati fuori timing: `inv[i]=1/sqrt(Omega[i,i])`, `A[i,j]=(Omega[i,j]*inv[i])*inv[j]`, `c_scaled[i]=inv[i]*c[i]`. I buffer nativi e quelli Eigen sono preparati separatamente da questi input; nessuna conversione Eigen-nativo e misurata o richiesta dal core. Eigen ColMajor riproduce il formato applicativo; Eigen RowMajor e il controllo con formato CSR analogo al nativo. Le differenze tra formati non sono attribuite a SIMD. Costruzione densa e assemblaggio sparso sono misurati separatamente e includono allocazione/distruzione del risultato. Il puntatore al risultato viene esposto a una barriera standard `atomic_signal_fence` prima della distruzione, uguale per tutti i backend, per conservare il lavoro di materializzazione.

Il replay FISTA e soltanto un harness nel benchmark. Tutti i backend condividono i loop scalari di proiezione/momentum e cambiano soltanto SpMV; non misura il vantaggio combinato di tutte le nuove API. `median_ns` misura la chiamata completa dell'harness con la verifica indipendente finale e la normalizzazione; `fista_timings_ns` misura separatamente reset/iterazioni/KKT periodici. Un'esecuzione separata strumentata restituisce tempo e numero di SpMV, overhead stimato dei timer e quota sul proprio tempo di iterazione strumentato. Tale quota non e la quota di SpMV nell'applicazione RGCCA.

Il certificato finale usa i triplet canonici indipendenti: KKT, non negativita, norma fisica, supporto e confronto con il peso catturato. Restart e iterazioni sono conservati, con concordanza dei supporti e distanza fra pesi dei backend. Il replay omette certificazione SPD mediante LDLT, gate diretto, fattorizzazioni dei supporti e pruning; i suoi conteggi possono differire dalla produzione. Non certifica scelta di lambda, pruning, o migrazione dell'applicazione.

Non si usa fast-math. Il core conserva l'ordine crescente dei termini di ogni output e il benchmark imposta `-ffp-contract=off`. Le riduzioni a pacchetti di Eigen possono avere un ordine diverso. La verifica compensata della norma fisica usa esplicitamente `std::fma`, fuori dal tempo di iterazione; questo non abilita contraction nei kernel. ISA, compilatore, clock, cache e carico modificano i risultati: ripetere su Kami in un nodo esclusivo prima delle conclusioni finali.

## Kami: job v2 sul nodo dedicato

Il launcher esistente `kami_simd.sh` ora esegue anche questo confronto. `submit` prenota un nodo esclusivo con PBS Pro (`select=1:ncpus=4:mem=32gb`, `place=excl`, queue `test`, walltime 72 ore di default). I quattro CPU servono alle build/test; tutte le misure usano un solo CPU fra quelli consentiti e thread interni a uno. Non interpretare PBS_NODEFILE come elenco dei CPU. Non eseguire lo sweep sul login.

Il job conserva i test nativi nelle quattro combinazioni dei flag, amplia i sanitizer alle API preallocate e al replay, prepara i binari dei vecchi sweep, poi esegue il confronto Eigen/preallocated e gli sweep precedenti. Build e misure non si sovrappongono. Il nuovo confronto conserva l'intero schedule, tutti i backend/API e 5 coppie alternate, 5 round e batch target 10 ms di default. Con le quattro capsule sono 143 casi e 575 confronti; senza capsule il job dichiara esplicitamente che ha soltanto sintetici.

`~/kami-vars.sh` e `~/kami-load.sh` vengono caricati in quest'ordine sul login e sul worker. Eigen **3.4.0** e i cinque file per ogni capsula sono verificati prima di qsub e nuovamente sul worker, senza installazioni. La sorgente GoogleTest viene predisposta una sola volta con `prepare` e riusata offline nel job. `prepare` resta indipendente da Eigen.

Dal checkout su Kami, dopo aver trasferito anche gli snapshot offline:

```bash
source ~/kami-vars.sh
source ~/kami-load.sh
export SIMD_CMAKE=/opt/mox/spack_v1/opt/spack/linux-rocky9-zen3/gcc-12.1.0/cmake-3.30.5-b7aplioxqel2hr44kp5zdsuzlo7ic23z/bin/cmake
export SIMD_CTEST="${SIMD_CMAKE%/*}/ctest"
export SIMD_EIGEN_INCLUDE="${PATH_EIGEN_INCLUDE:?caricare il profilo Eigen}"
export SIMD_REPLAY_MANIFEST="$PWD/output/simd/replay-inputs/manifest.json"
bash tests/benchmarks/kami_simd.sh prepare
bash tests/benchmarks/kami_simd.sh submit
```

`CXX` puo selezionare un compilatore C++20 gia installato; il launcher usa `g++` se non e impostato. I percorsi CMake/CTest sopra sono quelli Spack gia individuati, distinti dai programmi del login. Il job mantiene i percorsi assoluti scelti anche quando i profili del worker cambiano PATH. L'output va in una directory nuova `output/simd/kami/<timestamp>-<commit>/`, senza sovrascrivere il primo run.

Parametri del nuovo confronto: `SIMD_PREALLOCATED_PAIRS` (5), `SIMD_PREALLOCATED_ROUNDS` (5), `SIMD_PREALLOCATED_ROUND_MS` (10); `SIMD_TIMEOUT` vale per entrambi i runner. Restano disponibili `SIMD_CPUS`, `SIMD_QUEUE`, `SIMD_MEM`, `SIMD_WALLTIME`, i parametri degli sweep precedenti e `SIMD_KAMI_ENV_DIR` per profili in una directory diversa. `SIMD_EIGEN_INCLUDE` prevale su `PATH_EIGEN_INCLUDE`.

Il riepilogo unico e `<run>/summary.md`: include stato globale, test, sweep precedenti e il report completo Eigen/preallocated. `<run>/preallocated/` conserva manifest/hash, dati grezzi e summary Markdown/CSV/JSON; `<run>/preallocated-comparison.log` contiene compilazione/esecuzione del nuovo runner. Un fallimento, MISMATCH o output parziale resta visibile e non viene convertito in un confronto verificato. Fine schedule e plateau sono distinti: per i nuovi casi non si certifica automaticamente un plateau.

### Aggiornare il codice con Git

Codice, test e script sono disponibili nel branch `develop-SIMD`:

```bash
cd /u/donelli/Documents/fdaPDE-core-SIMD
git pull --ff-only origin develop-SIMD
```

Poi usare `prepare` e `submit` come sopra. I quattro snapshot offline non sono inclusi nel repository: il manifest predefinito resta `output/simd/replay-inputs/manifest.json`, oppure se ne puo scegliere uno con `SIMD_REPLAY_MANIFEST`. Il launcher usa soltanto `prefix_relative` e `copy_relative` sotto la directory del manifest, senza accedere al checkout RGCCA, e verifica SHA256 e dimensioni.

Un manifest esplicitamente richiesto ma assente o modificato fa fallire il preflight. Se il manifest predefinito e assente, il report dichiara un confronto soltanto sintetico; nessun replay reale viene dedotto. Le quattro capsule originali coprono componente 1, lambda 1/1000.

Il launcher non esegue SSH e non modifica RGCCA. `submit` invia il job; i risultati generati restano in `output/simd/kami/`.

### Verificare il launcher senza PBS o tempi locali

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/benchmarks/check_kami_environment.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/benchmarks/summarize_simd_sweep.py --self-test
```

Il check usa strumenti e runner simulati: verifica caricamento profili, percorsi Spack persistenti, Eigen, nodo esclusivo, passaggio delle capsule e dell'affinita, report completo e parziale e mancata sottomissione con input corrotti. Non compila il core, non misura kernel e non prenota nodi.

Per lanciare soltanto il nuovo runner dentro un job esclusivo gia allocato, scegliere un CPU consentito e passare i prefissi delle capsule con `--input`:

```bash
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
python3 tests/benchmarks/run_preallocated_comparison.py \
  --compiler "${CXX:-g++}" --eigen-include "$PATH_EIGEN_INCLUDE" \
  --cpu "$cpu" --output output/simd/eigen-comparison/kami \
  --pairs 5 --rounds 5 --round-ms 10 --timeout 900
```
