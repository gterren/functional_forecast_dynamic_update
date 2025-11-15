# validate_ffc.py wind observation 
for B in 72 144 216; do
  for A in forget_rate_f length_scale_f xi gamma kappa_min kappa_max; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# validate_ffc.py solar observation 
for B in 120 144 168; do
  for A in forget_rate_f length_scale_f xi gamma kappa_min kappa_max; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# validate_ffc.py wind dayahead 
for B in 72 144 216; do
  for A in forget_rate_e length_scale_e xi nu gamma kappa_min kappa_max; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# validate_ffc.py solar dayahead 
for B in 120 144 168; do
  for A in forget_rate_e length_scale_e xi nu gamma kappa_min kappa_max; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# validate_ffc.py solar fusion 
for B in 120 144 168; do
  for A in forget_rate_f forget_rate_e lookup_rate length_scale_f length_scale_e trust_rate nu xi gamma kappa_min kappa_max; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# validate_ffc.py wind fusion 
for B in 72 144 216; do
  for A in forget_rate_f forget_rate_e lookup_rate length_scale_f length_scale_e trust_rate nu xi gamma kappa_min kappa_max; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# envelop_ffc.py wind fusion 
for C in 0 1; do
  for B in 72 144; do
    for A in fknn l2; do
      sbatch run.job $A $B $C;
      sleep 5s
    done;
  done;
done;

# envelop_ffc.py wind fusion 
for C in 0 1 2 3; do
  for B in 72 144 216; do
    for A in fknn l2 sup; do
      sbatch run.job $A $B $C;
      sleep 5s
    done;
  done;
done;

# envelop_ffc.py solar fusion 
for C in 0 1 2 3; do
  for B in 120 144 168; do
    for A in fknn l2 sup; do
      sbatch run.job $A $B $C;
      sleep 5s
    done;
  done;
done;

# test_ffc.py solar observation/dayhead/fusion
for A in 120 144 168; do
  sbatch run.job $A;
  sleep 5s
done;

# test_ffc.py wind observation/dayhead/fusion
for A in 72 144 216; do
  sbatch run.job $A;
  sleep 5s
done;