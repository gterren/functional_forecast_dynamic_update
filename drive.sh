for A in forget_rate_f forget_rate_e length_scale_f length_scale_e; do
  for B in 144; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in forget_rate_f forget_rate_e lookup_rate length_scale_f length_scale_e trust_rate nu xi gamma kappa_min kappa_max; do
  for B in 120 144 168; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in forget_rate_f forget_rate_e lookup_rate length_scale_f length_scale_e trust_rate nu xi gamma kappa_min kappa_max; do
  for B in 72 144 216; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in 72 144; do
  for B in 0 1 2 3; do
    for C in fknn; do
      sbatch run.job $A $B $C;
      sleep 5s
    done;
  done;
done;

for A in 120 144 168; do
  for B in 0 1 2 3; do
    for C in fknn l2 sup; do
      sbatch run.job $A $B $C;
      sleep 5s
    done;
  done;
done;

for A in forget_rate_f length_scale_f gamma xi kappa_min kappa_max; do
  for B in 144; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;
