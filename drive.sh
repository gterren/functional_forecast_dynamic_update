for A in forget_rate_f forget_rate_e length_scale_f length_scale_e; do
  for B in 144; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in forget_rate_f forget_rate_e length_scale_f length_scale_e lookup_rate trust_rate nu gamma xi kappa_min kappa_max; do
  for B in 72 144 216; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in 72; do
  for B in zeta_1 zeta_2 zeta_3 zeta_4; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

for A in 144; do
  for B in alpha; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;