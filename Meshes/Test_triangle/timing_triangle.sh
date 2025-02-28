#!/bin/bash

gcc Meshes/Test_triangle/triangle.c -o Meshes/Test_triangle/triangle -lm

# Path al binario Triangle (compilato da triangle.c)
TRIANGLE="Meshes/Test_triangle/triangle"

# File di input
POLYFILE="Meshes/Test_triangle/rectangle_timing.poly"

# File CSV di output
OUTCSV="Meshes/Test_triangle/timing_triangle.csv"
echo "NumPoints,TimeElapsed(ms)" > "$OUTCSV"

# Array di valori di max_area da testare
domain_area=$((4000 * 2000))
divisors=(1000 10000 50000 100000)

AREAS=()
for d in "${divisors[@]}"; do
    area=$((domain_area / d))
    AREAS+=("$area")
done

# Loop su ciascun valore
for A in "${AREAS[@]}"; do

    # Rimuove file precedenti
    rm -f rectangle.1.*

    # Misura tempo iniziale (in nanosecondi)
    START=$(date +%s.%N)

    # Esegue triangle con opzioni: quality q=25°, area a=A
    $TRIANGLE -pq20a$A "$POLYFILE" > /dev/null

    # Misura tempo finale
    END=$(date +%s.%N)

    # Calcola tempo in ms (millisecondi)
    TIME_MS=$(awk "BEGIN {print ($END - $START)*1000}")

    # Conta i punti nel file .node generato
    if [[ -f Meshes/Test_triangle/rectangle_timing.1.node ]]; then
        NUMPTS=$(awk 'NR==1 {print $1}' Meshes/Test_triangle/rectangle_timing.1.node)
    else
        NUMPTS=0
    fi

    echo "$NUMPTS,$TIME_MS" >> "$OUTCSV"
done

python3 Meshes/Test_triangle/plot_timing.py
