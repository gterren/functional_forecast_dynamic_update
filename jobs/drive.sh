for B in 1 2 3 4 5; do
  for A in 120 144 168; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for B in 1 2 3 4 5; do
  for A in 72 144 216; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for B in 1 2 3 4 5; do
  for A in 6 12 18; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;