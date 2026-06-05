#!/bin/bash
# Shows GPU node availability across the cluster

echo "========================================"
echo "  CLUSTER GPU NODE STATUS"
echo "========================================"
echo ""

echo "--- Node Summary (State / GPUs / CPUs / Mem) ---"
sinfo -o "%-12N %-10P %-22G %-16C %-12m %-10t" | grep -v "^NODELIST" | sort -k6

echo ""
echo "--- Idle / Partially Free Nodes ---"
squeue --noheader -o "%N %b" | sort | uniq -c | sort -rn > /tmp/_used_gpus.tmp

sinfo -o "%N %G %C %m %t" --noheader | while read node gres cpus mem state; do
    total_gpus=$(echo "$gres" | grep -oP '(?<=gpu:)[^:]+:\K[0-9]+' | head -1)
    gpu_type=$(echo "$gres" | grep -oP '(?<=gpu:)[^:]+' | head -1)
    idle_cpus=$(echo "$cpus" | cut -d'/' -f2)
    case "$state" in
        idle|idle-)
            echo "  [IDLE]  $node  | GPUs: $total_gpus x $gpu_type | Idle CPUs: $idle_cpus | Mem: ${mem}MB"
            ;;
        mix|mix-)
            echo "  [PART]  $node  | GPUs: $total_gpus x $gpu_type | Idle CPUs: $idle_cpus | Mem: ${mem}MB"
            ;;
    esac
done

echo ""
echo "--- Running Jobs on GPU Nodes ---"
squeue -o "%-10i %-12u %-20j %-10T %-12l %-6D %-20R %-20b" | head -40

echo ""
echo "--- Your Jobs ---"
squeue -u "$USER" -o "%-10i %-20j %-10T %-12l %-20R %-20b"
