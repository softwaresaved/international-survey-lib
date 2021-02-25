for i in *.py; do
  if [ "$i" = "overview_and_sampling.py" ]; then
    continue
  fi
  python $i
done
