for A in 132; do
  for B in 1 2 3 4 5; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in 72 144 216; do
  for B in 1 2 3 4 5; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in 6 12 18; do
  for B in 1 2 3 4 5; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

