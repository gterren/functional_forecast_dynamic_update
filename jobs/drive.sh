# validate_ffc.py observation 
for A in forget_rate_f length_scale_f xi gamma kappa_min kappa_max; do
  sbatch run.job $A;
  sleep 5s
done;

# validate_ffc.py dayahead 
for A in forget_rate_e length_scale_e xi nu gamma kappa_min kappa_max; do
  sbatch run.job $A;
  sleep 5s
done;

# validate_ffc.py fusion
# for A in 72 144 216; do
for A in 72 144 216; do
  for B in forget_rate_f forget_rate_e lookup_rate length_scale_f length_scale_e trust_rate nu xi gamma kappa p_fusion; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# validate_envelop.py/test_envelop.py wind fusion 
for A in 72 144 216; do
  for B in fknn l2 sup; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# validate_envelop.py/test_envelop.py solar fusion 
for A in 120 144 168; do
  for B in fknn l2 sup; do
    sbatch run.job $A $B;
    sleep 5s
  done;
done;

# test_ffc.py solar observation/dayhead/fusion
for A in 120 144 168; do
  sbatch run.job $A;
  sleep 5s
done;

for A in 72 144 216; do
  sbatch run.job $A;
  sleep 5s
done;

# test_ffc.py wind observation/dayhead/fusion
for A in observation dayhead fusion; do
  for B in 72 144 216; do
    for C in fknn l2 sup; do
      sbatch run.job $A $B $C;
      sleep 5s
    done;
  done;
done;

# validate_ffc.py solar/wind fusion 
for A in forget_rate_f forget_rate_e lookup_rate length_scale_f length_scale_e trust_rate nu xi gamma kappa_min kappa_max; do
  sbatch run.job $A;
  sleep 5s
done;


