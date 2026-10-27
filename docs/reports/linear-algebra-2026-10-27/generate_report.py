"""Build the dated Italian report from archived benchmark and verification evidence."""
from pathlib import Path
import gzip
import json
import statistics
from collections import Counter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = HERE / 'evidence'
OUTPUT = ROOT / 'output/pdf/linear-algebra-optimizations-2026-10-27.pdf'


def archived(name):
    """Read an unchanged historical source from its compressed archive."""
    return json.loads(gzip.decompress((EVIDENCE / (name + '.gz')).read_bytes()))


def make_report():
    """Lay out separate historical results, current validation and reproduction instructions."""
    validation = json.loads((EVIDENCE / 'validation.json').read_text())
    current = json.loads((EVIDENCE / 'current-benchmarks.json').read_text())
    apple = archived('native-product-apple.json')
    clang = {x['case']: x for x in archived('native-product-clang17.json')}
    eigen = archived('eigen-comparisons.json')
    sweep = archived('sweep-v1.json')
    pdfmetrics.registerFont(TTFont('Report', '/System/Library/Fonts/Supplemental/Arial.ttf'))
    pdfmetrics.registerFont(TTFont('ReportBold', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
    navy, teal = colors.HexColor('#17334c'), colors.HexColor('#087f83')
    style = getSampleStyleSheet()
    style.add(ParagraphStyle(name='Main', fontName='ReportBold', fontSize=27, leading=31, textColor=navy, spaceAfter=16))
    style.add(ParagraphStyle(name='Section', fontName='ReportBold', fontSize=17, leading=21, textColor=navy, spaceAfter=13))
    style.add(ParagraphStyle(name='Copy', fontName='Report', fontSize=10, leading=14.3, spaceAfter=10))
    style.add(ParagraphStyle(name='SmallCopy', fontName='Report', fontSize=8.2, leading=11.2, spaceAfter=8))
    style.add(ParagraphStyle(name='Cell', fontName='Report', fontSize=8.1, leading=10.7))
    story = []

    def p(text, kind='Copy'):
        story.append(Paragraph(text, style[kind]))

    def title(text):
        p(text, 'Section')

    def table(headers, rows, widths):
        values = [[Paragraph(escape(str(v)), style['Cell']) for v in row] for row in [headers] + rows]
        t = Table(values, colWidths=[w * mm for w in widths], repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e3eef3')),
            ('TEXTCOLOR', (0, 0), (-1, 0), navy),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f8fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 0), (-1, 0), .6, teal)]))
        story.extend([t, Spacer(1, 11)])

    p('fdaPDE / CORE / 27 OTTOBRE 2026', 'SmallCopy')
    p('Ottimizzazioni<br/>di algebra lineare', 'Main')
    p('Integrazione di kernel, cache e batch paralleli su <b>develop-SIMD</b>. Report tecnico con risultati storici e verifiche ripetute sul codice aggiornato.')
    p('<b>Esito.</b> Il branch tematico mantiene i percorsi SIMD e sparsi preallocati e riceve il modulo di algebra strutturata usato da GFA: kernel SPD2/SPD3, cache selettive e operazioni batch con esecuzione parallela esplicita.')
    table(['Riferimento', 'Revisione / ruolo'], [
        ['Branch aggiornato', 'develop-SIMD'],
        ['Base SIMD', 'f582eb46 - snapshot precedente all\'integrazione'],
        ['Sorgente GFA', 'f146ddf2 - develop-GFA, solo algebra e prerequisiti execution'],
        ['Codice verificato', validation['code_revision'][:12]],
        ['Ambiente locale', 'Apple M3 Max, macOS 26.6.2; AppleClang 21 e GCC 15'],
    ], [42, 128])
    p('<b>Tre meccanismi distinti.</b> SIMD espone coefficienti contigui al vettorizzatore del compilatore; i kernel specializzati riducono il lavoro su matrici piccole; execution_par distribuisce elementi batch indipendenti su più thread. La presenza di uno non attiva automaticamente gli altri.')
    p('Il progetto GFA usa già MatrixBatch, cache e execution_par nei driver di interpolazione, smoothing e penalty-consistency. La configurazione locale individua per impostazione predefinita il core condiviso su develop-GFA. I flag SIMD appartengono al branch tematico e restano opt-in.')
    p('Le modifiche C-LE non committate nella copia GFA-core-cle-spd3 non fanno parte del trasferimento. Non sono stati importati i solver geometrici o l\'applicazione GFA.', 'SmallCopy')
    story.append(PageBreak())

    title('1 / Modulo aggiornato e contratti')
    table(['Componente', 'Aggiornamento trasferito'], [
        ['SIMD denso', 'Assegnamenti contigui, prodotti float/double con accumulo contiguo e API preallocate conservati'],
        ['SPD2 / SPD3', 'EVD a dimensione fissa, candidati certificati e fallback Jacobi scalato; funzioni spettrali e derivate'],
        ['Cache strutturate', 'Politiche selettive per spettro, logaritmo, radici e differenze divise; riuso durante espansione di policy'],
        ['MatrixBatch', 'Storage aggregato, selezioni, aritmetica lazy fusa, trasformazioni proprietarie sequenziali o parallele'],
        ['API algebra', 'Cholesky, rotazioni e quaternioni; inv(), trace(), orthogonalize(); determinante da fattori LU già calcolati'],
        ['Sparso', 'API CSR/preallocate SIMD mantenute; correzioni Eigen SparseBlockMatrix e multigrid a ciclo fisso aggiunti'],
        ['Execution', 'Fondazione con ownership e contabilità dei task; parallel_for con propagazione delle eccezioni'],
    ], [36, 134])
    p('<b>Materializzazione.</b> La costruzione di Matrix riconosce prima gli oggetti con eval_matrix(); per i prodotti ordinari continua a selezionare il percorso contiguo SIMD. Questo punto richiedeva una risoluzione esplicita del conflitto tra le due linee di sviluppo.')
    p('<b>Cache e alias.</b> SPD mantiene l\'invariante di positività mediante sostituzioni verificate. Le scritture tracciate invalidano le cache; un puntatore raw mutabile disabilita il riuso per quel binding. Le assegnazioni pubbliche conservano le garanzie sugli alias. assign_disjoint richiede storage non sovrapposto.')
    p('<b>Parallelismo.</b> Le operazioni restano sequenziali per impostazione predefinita. Il chiamante sceglie execution_par e configura i thread prima dell\'inizializzazione del runtime. Le riduzioni batch restano sequenziali. Le eccezioni di parallel_for vengono riportate dopo il completamento del lavoro gia pubblicato.')
    p('<b>Interfacce.</b> dense_linear_algebra.h espone la parte nativa senza Eigen; sparse_linear_algebra.h mantiene lo sparso nativo; linear_algebra.h compone inoltre gli strumenti dipendenti da Eigen. I tutorial numerati 01-11 e la documentazione dei contratti sono aggiornati.')
    p('Provenienza: le basi execution e SPD spectral sono integrate mantenendo gli autori originali. Il port successivo cita esplicitamente la revisione GFA e il perimetro parziale; non include coautori automatici.', 'SmallCopy')
    story.append(PageBreak())

    title('2 / Benchmark SIMD storici')
    p('Misure archiviate precedenti al port, su M3 Max. Il riferimento dei prodotti è 91be5d5. La chiamata pubblica comprende allocazione, temporanei e copie; setup e oracolo sono esclusi. Assertion disabilitate nei benchmark, ffp-contract=off. Il rapporto è tempo prima / tempo dopo: valori maggiori di 1 indicano un miglioramento.')
    rows = []
    for x in apple:
        rows.append([x['case'], x['scalar'], ' x '.join(map(str, x['shape'])), f"{x['baseline_ns']/1000:.3f}", f"{x['current_ns']/1000:.3f}", f"{x['ratio']:.2f}x", f"{clang[x['case']]['ratio']:.2f}x"])
    table(['Caso', 'Tipo', 'm x k x n', 'Prima us', 'Dopo us', 'Apple 21', 'Clang 17'], rows, [30, 16, 34, 23, 23, 22, 22])
    p('Il caso 128 x 128 double migliora di circa 4,1x e quello float di circa 8,5x nella campagna AppleClang. I casi piccoli e i layout non eleggibili mostrano benefici modesti o nessun beneficio. Queste misure non descrivono il nuovo port e non costituiscono una garanzia sulle altre CPU.', 'SmallCopy')
    v1 = Counter(x['suite'] for x in sweep)
    p(f'<b>Sweep V1.</b> {len(sweep)} righe archiviate: 314 assignment primarie, 66 product primarie e 30 factorial. Copertura primaria: 24 casi assignment, 6 dei 13 product; mixed_col non ha stop registrato. Nessun plateau confermato.')
    v2 = archived('sweep-v2.json')
    p(f'<b>V2 rimane parziale.</b> Sono presenti {len(v2)} righe, solo square_row e square_col; un solo stop, per limite di durata. Mancano i riepiloghi delle campagne large-assignment e Clang17-v2. I dati V1 e V2 restano separati.', 'SmallCopy')
    p('<b>Test storici associati.</b> Lo sweep riporta 114/114 test per AppleClang e GCC, OFF e ON. La successiva verifica preallocata GCC ON riporta 132/132; ASan/UBSan AppleClang verifica 7 test densi e 11 sparsi/replay. Sono risultati delle revisioni archiviate, distinti dalle nuove esecuzioni della sezione 4.', 'SmallCopy')
    story.append(PageBreak())

    title('3 / Confronti storici con Eigen 3.4')
    p('Campagna del 2 ottobre: baseline Git e14d9e41, header e input effettivi identificati tramite hash nel manifest originale. Sono archiviati 575 confronti verificati su 143 casi. Nativo SIMD OFF/ON e backend Eigen sono confrontati mediante processi abbinati. Le misure sono in background/nice19 su Darwin, senza garanzia di macchina quieta.')
    counts = Counter((x['case']['op'], x['case'].get('api', '')) for x in eigen)
    grouped = []
    for op in sorted({key[0] for key in counts}):
        entries = [(api, n) for (operation, api), n in sorted(counts.items()) if operation == op]
        grouped.append([op, '; '.join(f'{api}: {n}' for api, n in entries), sum(n for _, n in entries)])
    table(['Operazione', 'Ripartizione per API', 'Confronti'], grouped, [41, 96, 33])
    selected = []
    for x in eigen:
        c = x['case']
        if x['comparison'] == 'off/native:on/native' and c.get('api') == 'preallocated' and c['op'] in ('gemv', 'gemvt', 'spmv'):
            selected.append(x)
    selected = sorted(selected, key=lambda x: x['working_set_bytes'], reverse=True)
    used, rows = set(), []
    for x in selected:
        c = x['case']; key = (c['op'], c['rows'], c['cols'])
        if key in used: continue
        used.add(key)
        rows.append([c['op'], f"{c['rows']} x {c['cols']}", f"{x['off_ns']/1000:.2f}", f"{x['on_ns']/1000:.2f}", f"{x['ratio']:.3f} [{x['ratio_min']:.3f}, {x['ratio_max']:.3f}]"])
        if len(rows) == 6: break
    table(['Esempi preallocati', 'Forma', 'OFF us', 'ON us', 'Rapporto [min, max]'], rows, [32, 36, 24, 24, 54])
    p('La selezione mostra i maggiori working set disponibili per le operazioni elencate, non i casi migliori. I rapporti vicini o inferiori a 1 mostrano che il flag SIMD non produce un guadagno uniforme. Min/max esprime dispersione tra coppie di processi, non un intervallo di confidenza.', 'SmallCopy')
    p('<b>Allocazioni.</b> I percorsi preallocati nativi misurati non eseguono nuove allocazioni per chiamata disgiunta; le chiamate pubbliche possono includere snapshot. Il contatore C++ new non misura malloc di Eigen. Il confronto pubblico/preallocato cambia quindi anche il costo delle allocazioni.')
    p('<b>Replay RGCCA.</b> Copre il solo harness FISTA, con certificato finale e controlli distinti su input, supporto e pesi. Non dimostra prestazioni dell\'intera applicazione, pruning o fattorizzazioni di supporto. I dati densi alle forme applicative sono sintetici, senza matrice X catturata.')
    p('Le 575 righe complete sono in evidence/eigen-comparisons.csv; i riepiloghi JSON con coppie e round, il manifest e il controllo degli hash sono archiviati senza modifiche in formato gzip.', 'SmallCopy')
    story.append(PageBreak())

    title('4 / Verifica del codice aggiornato')
    p('Esecuzioni del 27 ottobre 2026. I build ordinari usano Release e warning come errori; le assertion fdaPDE restano attive. Le sole eccezioni NoDebug sono i test mirati di cancellazione degli argomenti delle assertion e ortogonalizzazione obbligatoria.')
    rows=[]
    for x in validation['lanes']:
        rows.append([x['name'], x['compiler'], x['simd'], f"{x['passed']}/{x['tests']}", x['negative_programs']])
    table(['Configurazione', 'Compilatore', 'SIMD', 'CTest', 'Reject'], rows, [47, 36, 17, 30, 40])
    p('Reject indica i programmi che devono essere rifiutati con lo specifico diagnostic previsto; non sono conteggiati tra i test CTest. La lane nativa verifica in configurazione che Eigen/Eigen non sia disponibile al compilatore.')
    table(['Copertura', 'Oracolo / proprieta verificata'], [
        ['SPD2 / SPD3', 'EVD dinamica e valori indipendenti a 80 cifre; casi quasi degeneri, scale estreme e autovalori ripetuti'],
        ['Funzioni spettrali', 'Ricostruzione e derivate di Frechet contro differenze finite centrali'],
        ['Cache e batch', 'Invalidazione, policy, alias, temporanei, ordine dei risultati, errori e riutilizzabilita del runtime'],
        ['SIMD e sparso', 'Oracoli scalari, layout e forme irregolari, buffer disallineati, preallocazione e alias'],
        ['Execution', 'Code concorrenti, accounting, nested parallelism, riduzioni, propagazione delle eccezioni'],
        ['Header / integrazione', 'Aggregate autonomi, ordini di inclusione e runtime condiviso tra translation unit'],
    ], [36, 134])
    p('<b>Correzioni di integrazione.</b> Aggiunte parentesi nel test di accuratezza SO(3) per il diagnostic dangling-else di GCC. Nel costruttore da coordinate simmetriche la copia usa la capacità della rappresentazione di destinazione, dopo la validazione del conteggio: evita il diagnostic array-bounds di GCC sul test che rifiuta un ordine incompatibile. Il test multigrid inizializza esplicitamente la matrice prima di riempirne le colonne, eliminando un diagnostic Eigen maybe-uninitialized senza sopprimere warning.')
    p('I risultati XML e i riepiloghi sono archiviati in evidence/. I tutorial numerati sono compilati ed eseguiti separatamente. Le verifiche di correttezza non costituiscono una prova formale del modello di memoria del runtime.', 'SmallCopy')
    story.append(PageBreak())

    title('5 / Misure attuali e riproducibilità')
    p('Probe sul codice integrato con AppleClang 21, Release, assertion attive. Setup escluso, allocazione e materializzazione dell\'output incluse. Per SPD3: 7 campioni di 3 chiamate complete, mediana per chiamata, 4 thread configurati. Ogni coefficiente sequenziale/parallelo e confrontato prima del timing.')
    times=current['timings_ns']
    rows=[]
    for count in (4096,65536):
        a=times[f'spd3_log_sequential_{count}'];b=times[f'spd3_log_parallel4_{count}']
        rows.append([count,f'{a/1e6:.3f}',f'{b/1e6:.3f}',f'{a/b:.2f}x'])
    table(['Matrici SPD3', 'Sequenziale ms', '4 thread ms', 'Rapporto'],rows,[40,46,46,38])
    p('La misura è locale, eseguita senza build lanciate da questa chat in parallelo. Non certifica l\'assenza di altri carichi sul computer; l\'ordine sequenziale poi parallelo non fornisce coppie randomizzate o un intervallo di confidenza. Non viene estrapolata all\'intera applicazione GFA.', 'SmallCopy')
    allocations=current['allocations']
    rows=[]
    for name in ['batch_static_none_identity','batch_static_full_identity','batch_dynamic_none_identity','batch_dynamic_full_identity','batch_dynamic_fused_arithmetic','batch_dynamic_mapped_arithmetic']:
        x=next(x for x in allocations if x['name']==name and (x['count']==1024 or 'arithmetic' in name))
        rows.append([name.replace('batch_',''),x['count'],x['persistent'],x['temporary'],x['total']])
    table(['Costruzione', 'Elementi', 'Persistenti', 'Temporanee', 'Totali'],rows,[74,24,24,24,24])
    p('Le allocazioni aggregate non crescono per elemento nei batch identity. L\'aritmetica fusa mantiene il numero di allocazioni della singola espressione nativa mappata; questo è un controllo sul lavoro evitato, separato dai rapporti di tempo.')
    pairs=json.loads((EVIDENCE/'current-simd-pairs.json').read_text())
    rows=[]
    for case in sorted({x['case'] for x in pairs}):
        selected=[x for x in pairs if x['case']==case]
        ratios=[x['ratio'] for x in selected]
        rows.append([case, selected[0]['size'], f'{statistics.median(ratios):.2f}x', f'{min(ratios):.2f} - {max(ratios):.2f}'])
    p('<b>Controllo SIMD attuale.</b> Tre coppie OFF/ON alternate, cinque round per processo, ffp-contract=off e assertion disabilitate soltanto nel benchmark. Oracolo completo attivo; dispersione min/max tra coppie, senza intervallo di confidenza.', 'SmallCopy')
    table(['Caso', 'Size del driver', 'Rapporto mediano', 'Min - max'], rows, [46, 40, 44, 40])
    p('<b>Riproduzione.</b> Dalla radice del core: configurare tests/ con CMake, compilare e lanciare CTest. Impostare FDAPDE_ENABLE_SIMD=1 per la lane ON, FDAPDE_NATIVE_ONLY=ON per l\'isolamento da Eigen e FDAPDE_BUILD_SPD_BENCHMARK=ON per il probe. Comandi esatti, versioni, hash e conteggi sono in evidence/validation.json e README.md di questa cartella.')
    p('<b>Limiti.</b> Nessuna esecuzione Kami/Linux aggiuntiva; nessuna misura end-to-end GFA nuova. I dati storici rimangono associati ai loro manifest. SIMD resta opt-in; nessun fast-math, nuova dipendenza di algebra o tuning universale della CPU.', 'SmallCopy')

    def page(canvas, doc):
        """Draw consistent page identification and a light footer rule."""
        canvas.setStrokeColor(colors.HexColor('#c8d8e2'))
        canvas.line(20*mm, 17*mm, 190*mm, 17*mm)
        canvas.setFont('Report',8)
        canvas.setFillColor(navy)
        canvas.drawString(20*mm,12*mm,'fdaPDE | develop-SIMD | 27 ottobre 2026')
        canvas.drawRightString(190*mm,12*mm,str(doc.page))

    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUTPUT),pagesize=(210*mm,297*mm),rightMargin=20*mm,leftMargin=20*mm,topMargin=20*mm,bottomMargin=23*mm,title='Ottimizzazioni di algebra lineare - fdaPDE',author='Pietro Donelli').build(story,onFirstPage=page,onLaterPages=page)
    print(OUTPUT)


if __name__=='__main__':
    make_report()
