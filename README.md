# Experimental evaluation scripts for the paper "Beyond Eager Encodings: A Theory-Agnostic Approach to Theory-Lemma Enumeration in SMT"

## Dependencies

```bash
# Install dependencies
$ pip3 install .

# Install MathSAT via PySMT
$ pysmt-install --msat

# Install TabularAllSAT solver
$ git clone https://github.com/giuspek/tabularAllSAT.git
$ cd tabularAllSAT/cdcl-vsads
$ ./configure
$ make
$ export TABULARALLSAT_PATH=$(pwd)/solver
```

## Benchmark layout

Benchmark inputs use one directory per instance:

```text
data/benchmark/<set>/<...>/<instance>/problem.smt2
data/benchmark/<set>/<...>/<instance>/queries/*.smt2  # optional
```

Generated outputs mirror the instance directory directly:

```text
results/<run>/<set>/<...>/<instance>/tlemmas.smt2
results/<run>/<set>/<...>/<instance>/logs.json
```

`tlemmas_gen` automatically uses the sibling `queries/` directory when present.
The old `tlemmas_gen_queries` command is not supported.

## How to run

### Generating T-lemmas

1. Create a config.yaml file.

   ```bash
   cp config.yaml.example config.yaml
   ```

2. Configure it according to the benchmark you want to execute.
   Here is an example for Planning problems, using 45 cores for parallelized
   solvers, timeouts set to 1h, and 16GB of RAM. You can leave `tlemmas_dir` and
   `gt_tlemmas_dir` empty.

   ```yaml
   # MB of memory per core
   memory: 16384
   # Maximum allowed solving time per problem in seconds
   timeout: 3600
   # Number of concurrent evaluations running
   processes: 1
   # For parallel t-lemma enumeration, number of concurrent AllSMT processes
   allsmt_processes: 45
   # Benchmarks
   benchmarks: [
       # Base test cases
       "data/benchmark/planning/h3/Painter",
     ]
   # T-Lemmas base path
   tlemmas_dir: ""
   # Ground truth logs path
   gt_tlemmas_dir: ""
   # Output dir, add a / to the end
   results: "results/"
   ```

3. Run the benchmark controller. For example:

   - Enumeration with sequential algorithm (baseline)

     ```bash
     $ python3 scripts/benchmark_controller.py <output_folder> \
     tlemmas_gen --solver sequential

     ```

   - Enumeration with parallel algorithm (divide&conquer)

     ```bash
     $ python3 scripts/benchmark_controller.py <output_folder> \
     tlemmas_gen --solver parallel
     ```

   - Enumeration with parallel algorithm with projection on T-atoms

     ```bash
     $ python3 scripts/benchmark_controller.py <output_folder> \
     tlemmas_gen --solver parallel --projection
     ```

   - Enumeration with parallel algorithm with projection on T-atoms \
      and partitioning

     ```bash
     $ python3 scripts/benchmark_controller.py <output_folder> \
     tlemmas_gen --solver parallel --projection --partition
     ```

### Checking T-lemmas

1. Fill the `tlemmas_dir` in `config.yaml` and `gt_tlemmas_dir` fields.
   The first is used to specify the folder in which the generated T-lemmas to
   check are stored. The second one is optional and is used to provide the ground
   truth data, as an additional check.

2. Run the benchmark controller:

   ```bash
   python3 scripts/benchmark_controller.py <output_folder> tlemmas_check
   ```
