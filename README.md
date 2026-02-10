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

## How to run

### Generating T-lemmas

First of all, you need to create a config.yaml file.
```bash
$ cp config.yaml.example config.yaml
```

Then, you need to configure it according to the benchmark you want to execute.
Here is an example for Planning problems, using 45 cores for parallelized solvers, timeouts set to 1h, and 16GB of RAM. You can leave `tlemmas_dir` and `gt_tlemmas_dir` empty.

```yaml
# MB of memory per core
memory: 16384

# Maximum allowed solving time per problem in seconds
timeout: 3600

# Number of concurrent evaluations running
processes: 1

# For compilation tasks, number of concurrent AllSMT processes per task that enumerates
# total truth assignments
allsmt_processes: 45

# Benchmarks
benchmarks: [
  # Base test cases
  "data/benchmark/planning/h3/Painter"
] 

# T-Lemmas base path
tlemmas_dir: ""

# Ground truth logs path
gt_tlemmas_dir: ""

# Output dir, add a / to the end
results: "results/"
```

Then, run the benchmark controller:
```bash
# Run with projection and partitioning
$ python3 scripts/benchmark_controller.py tlemmas_proj <output_folder> partition

# OR
# Run with projection alone
$ python3 scripts/benchmark_controller.py tlemmas_proj <output_folder> parallel

# OR
# Run with divide and conquer alone
$ python3 scripts/benchmark_controller.py tlemmas <output_folder> parallel

# OR
# Run with allsmt
$ python3 scripts/benchmark_controller.py tlemmas <output_folder> sequential
```

### Checking T-lemmas

In your `config.yaml`, you need to fill the `tlemmas_dir` and `gt_tlemmas_dir` fields.
The first is used to specify the folder in which the generated T-lemmas to check are stored.
The second one is optional and is used to provide the ground truth data, as an additional check. 

Then, run the benchmark controller:
```bash
$ python3 scripts/benchmark_controller.py tlemmas_check <output_folder> partition
```