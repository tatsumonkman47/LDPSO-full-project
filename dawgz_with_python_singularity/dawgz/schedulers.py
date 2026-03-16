r"""Scheduling backends"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import csv
import os
import shutil
import subprocess
import uuid

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache, partial
from inspect import isawaitable
from pathlib import Path
from random import random
from tabulate import tabulate
from typing import Any, Callable, Dict, Sequence

from .utils import cat, comma_separated, future, pickle, runpickle, slugify, trace
from .workflow import Job, cycles
from .workflow import prune as _prune

DIR = os.environ.get("DAWGZ_DIR", ".dawgz")
DIR = Path(DIR).resolve()


class Scheduler(ABC):
    r"""Abstract workflow scheduler."""

    backend: str = None

    def __init__(
        self,
        name: str = None,
        settings: Dict[str, Any] = {},  # noqa: B006
        **kwargs,
    ):
        r"""
        Arguments:
            name: The name of the workflow.
            settings: A dictionnary of settings.
            kwargs: Keyword arguments added to `settings`.
        """

        super().__init__()

        self.name = name
        self.date = datetime.now().replace(microsecond=0)
        self.uuid = uuid.uuid4().hex

        self.path = DIR / self.uuid
        self.path.mkdir(parents=True)

        # Settings
        self.settings = settings.copy()
        self.settings.update(kwargs)

        # Jobs
        self.order = {}
        self.results = {}
        self.traces = {}

    def dump(self):
        with open(self.path / "dump.pkl", "wb") as f:
            pickle.dump(self, f)

        with open(self.path.parent / "workflows.csv", mode="a", newline="") as f:
            csv.writer(f).writerow((
                self.name,
                self.uuid,
                self.date,
                self.backend,
                len(self.order),
                len(self.traces),
            ))

    @staticmethod
    def load(path: Path) -> Scheduler:
        with open(path / "dump.pkl", "rb") as f:
            return pickle.load(f)

    def tag(self, job: Job) -> str:
        if job in self.order:
            i = self.order[job]
        else:
            i = self.order[job] = len(self.order)

        return f"{i:04d}_{slugify(job.name)}"

    def state(self, job: Job, i: int = None) -> str:
        if job in self.traces:
            return "FAILED"
        else:
            return "COMPLETED"

    def output(self, job: Job, i: int = None) -> Any:
        if job.array is None:
            return self.results[job]
        else:
            return self.results[job].get(i)

    def report(self, job: Job = None) -> str:
        if job is None:
            headers = ("Name", "State")
            rows = [(str(job), self.state(job)) for job in self.order]

            return tabulate(rows, headers, showindex=True)
        else:
            headers = ("Name", "State", "Output")
            array = [None]

            if job in self.traces:
                rows = [(str(job), self.state(job), self.traces[job])]
            elif job.array is None:
                rows = [(str(job), self.state(job), self.output(job))]
            else:
                array = sorted(job.array)
                rows = [
                    (f"{job.name}[{i}]", self.state(job, i), self.output(job, i)) for i in array
                ]

            rows = [
                (
                    name,
                    state,
                    None if output is None else cat(output, width=120),
                )
                for name, state, output in rows
            ]

            return tabulate(rows, headers, showindex=array)

    def cancel(self, job: Job = None) -> str:
        raise NotImplementedError(f"'cancel' is not implemented for the {self.backend} backend.")

    @contextmanager
    def context(self):
        try:
            yield None
        finally:
            pass

    def __call__(self, *jobs: Job, prune: bool = False):
        for cycle in cycles(*jobs, backward=True):
            raise CyclicDependencyGraphError(" <- ".join(map(str, cycle)))

        if prune:
            jobs = _prune(*jobs)

        with self.context():
            asyncio.run(self.wait(*jobs))

    async def wait(self, *jobs: Job):
        if jobs:
            await asyncio.wait(map(asyncio.create_task, map(self.submit, jobs)))
            await asyncio.wait(map(asyncio.create_task, map(self.submit, self.order)))

    async def submit(self, job: Job) -> Any:
        if job in self.results:
            result = self.results[job]
        else:
            result = self.results[job] = future(self._submit(job), return_exceptions=True)

        if isawaitable(result):
            result = self.results[job] = await result

            if isinstance(result, Exception):
                self.traces[job] = trace(result)

        return result

    async def _submit(self, job: Job) -> Any:
        try:
            if job.satisfiable:
                await self.satisfy(job)
            else:
                raise DependencyNeverSatisfiedError(str(job))
        finally:
            self.tag(job)

        return await self.exec(job)

    @abstractmethod
    async def satisfy(self, job: Job):
        pass

    @abstractmethod
    async def exec(self, job: Job) -> Any:
        pass


class AsyncScheduler(Scheduler):
    r"""Asynchronous scheduler.

    Jobs are executed asynchronously. A job is launched as soon as its dependencies are
    satisfied.
    """

    backend: str = "async"

    def __init__(self, name: str = None, pools: int = None, **kwargs):
        r"""
        Arguments:
            name: The name of the workflow.
            pools: The number of processing pools. If `None`, use threads instead.
            kwargs: Keyword arguments passed to :class:`Scheduler`.
        """

        super().__init__(name=name, **kwargs)

        self.pools = pools

    @contextmanager
    def context(self):
        if self.pools is None:
            self.executor = cf.ThreadPoolExecutor()
        else:
            self.executor = cf.ProcessPoolExecutor(self.pools)

        try:
            yield None
        finally:
            del self.executor

    async def satisfy(self, job: Job):
        pending = [
            asyncio.gather(self.submit(dep), future(status))
            for dep, status in job.dependencies.items()
        ]

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                result, status = task.result()

                if isinstance(result, JobFailedError) and status != "success":
                    result = None
                elif not isinstance(result, Exception) and status == "failure":
                    result = JobNotFailedError(f"{job}")

                if isinstance(result, Exception):
                    if job.waitfor == "all":
                        raise DependencyNeverSatisfiedError(str(job)) from result
                elif job.waitfor == "any":
                    break
            else:
                continue
            break
        else:
            if job.dependencies and job.waitfor == "any":
                raise DependencyNeverSatisfiedError(str(job))

    async def exec(self, job: Job) -> Any:
        dump = pickle.dumps(job.run)
        call = partial(self.remote, runpickle, dump)

        try:
            if job.array is None:
                return await call()
            else:
                results = await asyncio.gather(*map(call, job.array), return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        raise result

                return dict(zip(job.array, results))
        except Exception as e:
            raise JobFailedError(str(job)) from e

    async def remote(self, f: Callable, /, *args) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self.executor, f, *args)


class DummyScheduler(AsyncScheduler):
    r"""Dummy asynchronous scheduler.

    Jobs are scheduled asynchronously, but instead of executing them, their name is
    printed before and after a short (random) sleep time. Useful for debugging.
    """

    backend: str = "dummy"

    async def exec(self, job: Job):
        print(f"START {job}")
        await asyncio.sleep(random())
        print(f"END   {job}")

        return None if job.array is None else {}


class SlurmScheduler(Scheduler):
    r"""Slurm scheduler.

    Jobs are submitted to the Slurm queue. Resources are allocated by the Slurm manager
    according to the job and scheduler settings. Job settings have precendence over
    scheduler settings.

    Most settings (e.g. `account`, `export`, `partition`) are passed directly to
    `sbatch`. A few settings (e.g. `cpus`, `gpus`, `ram`) are translated into their
    `sbatch` equivalents.
    """

    backend: str = "slurm"
    translate: Dict[str, str] = {
        "cpus": "cpus-per-task",
        "gpus": "gpus-per-task",
        "ram": "mem",
        "memory": "mem",
        "timelimit": "time",
    }

    def __init__(
        self,
        name: str = None,
        shell: str = os.environ.get("SHELL", "/bin/sh"),
        interpreter: str = "python",
        env: Sequence[str] = [],  # noqa: B006
        **kwargs,
    ):
        r"""
        Arguments:
            name: The name of the workflow.
            shell: The scripting shell.
            interpreter: The Python interpreter.
            env: A sequence of commands to execute before each job is launched.
            kwargs: Keyword arguments passed to :class:`Scheduler`.
        """

        super().__init__(name=name, **kwargs)

        assert shutil.which("sbatch") is not None, "sbatch executable not found"

        # Environment
        self.shell = shell
        self.interpreter = interpreter
        self.env = env

    @lru_cache(None)  # noqa: B019
    def sacct(self, jobid: str) -> Dict[str, str]:
        text = subprocess.run(
            ["sacct", "-j", jobid, "-o", "JobID,State", "-n", "-P", "-X"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout

        return dict(line.split("|") for line in text.splitlines())

    def state(self, job: Job, i: int = None) -> str:
        if job in self.traces:
            return "CANCELLED"

        jobid = self.results[job]
        table = self.sacct(jobid)

        if job.array is None:
            return table.get(jobid, None)
        elif i is None:
            if table:
                return ",".join(sorted(set(table.values())))
            else:
                return None
        elif i in job.array:
            return table.get(f"{jobid}_{i}", None)
        else:
            return None

    def output(self, job: Job, i: int = None) -> str:
        tag = self.tag(job)

        if job.array is None:
            logfile = self.path / f"{tag}.log"
        else:
            logfile = self.path / f"{tag}_{i}.log"

        if logfile.exists():
            with open(logfile, newline="", errors="replace") as f:
                return f.read()
        else:
            return None

    def report(self, job: Job = None) -> str:
        if job is None:
            headers = ("Name", "ID", "State")
            rows = []

            for job in self.order:
                if job in self.traces:
                    jobid = None
                else:
                    jobid = self.results[job]

                rows.append((str(job), jobid, self.state(job)))

            return tabulate(rows, headers, showindex=True)
        else:
            return super().report(job)

    def cancel(self, job: Job = None) -> str:
        if job is None:
            jobids = list(self.results.values())
        else:
            jobid = self.results[job]
            jobids = [jobid]

        return subprocess.run(
            ["scancel", "-v", *jobids],
            capture_output=True,
            check=True,
            text=True,
        ).stderr.strip("\n")

    async def satisfy(self, job: Job) -> str:
        results = await asyncio.gather(*map(self.submit, job.dependencies))

        for result in results:
            if isinstance(result, Exception):
                raise DependencyNeverSatisfiedError(str(job)) from result

    async def exec(self, job: Job) -> Any:
        # Submission script
        lines = [
            f"#!{self.shell}",
            "#",
            f'#SBATCH --job-name="{job.name}"',
        ]

        if job.array is not None:
            indices = comma_separated(job.array)

            if job.array_throttle is None:
                lines.append(f"#SBATCH --array={indices}")
            else:
                lines.append(f"#SBATCH --array={indices}%{job.array_throttle}")

        tag = self.tag(job)

        if job.array is None:
            logfile = self.path / f"{tag}.log"
        else:
            logfile = self.path / f"{tag}_%a.log"

        lines.append(f"#SBATCH --output={logfile}")

        ## Settings
        settings = self.settings.copy()
        settings.update(job.settings)

        assert "clusters" not in settings, "multi-cluster jobs not supported"

        for key in settings:
            assert not key.startswith("ntasks"), "multi-task jobs not supported"

        nodes = settings.pop("nodes", 1)

        lines.append("#")
        lines.append("#SBATCH --nodes=" + f"{nodes}")
        lines.append("#SBATCH --ntasks-per-node=1")

        for key, value in settings.items():
            key = self.translate.get(key, key)

            if type(value) is bool:
                if value:
                    lines.append(f"#SBATCH --{key}")
            else:
                lines.append(f"#SBATCH --{key}={value}")

        ## Dependencies
        sep = "?" if job.waitfor == "any" else ","
        types = {
            "success": "afterok",
            "failure": "afternotok",
            "any": "afterany",
        }

        deps = [
            f"{types[status]}:{await self.submit(dep)}" for dep, status in job.dependencies.items()
        ]

        if deps:
            lines.append("#")
            lines.append("#SBATCH --dependency=" + sep.join(deps))

        lines.append("")

        ## Environment
        if self.env:
            lines.extend([*self.env, ""])

        ## Pickle job
        pklfile = self.path / f"{tag}.pkl"

        with open(pklfile, "wb") as f:
            pickle.dump(job.run, f)

        pyfile = self.path / f"{tag}.py"

        with open(pyfile, "w") as f:
            f.write(
                "\n".join([
                    "import argparse",
                    "import pickle",
                    "",
                    "parser = argparse.ArgumentParser()",
                    "parser.add_argument('-i', '--index', type=int, default=None)",
                    "",
                    "args = parser.parse_args()",
                    "",
                    "with open('{}', 'rb') as f:".format(pklfile),
                    "    if args.index is None:",
                    "        pickle.load(f)()",
                    "    else:",
                    "        pickle.load(f)(args.index)",
                    "",
                ])
            )

        if job.interpreter is None:
            interpreter = self.interpreter
        else:
            interpreter = job.interpreter

        if job.array is None:
            lines.append(f"srun {interpreter} {pyfile}")
        else:
            lines.append(f"srun {interpreter} {pyfile} -i $SLURM_ARRAY_TASK_ID")

        lines.append("")

        ## Save
        shfile = self.path / f"{tag}.sh"

        with open(shfile, "w") as f:
            f.write("\n".join(lines))

        # Submit script
        try:
            text = subprocess.run(
                ["sbatch", "--parsable", str(shfile)],
                capture_output=True,
                check=True,
                text=True,
            ).stdout

            jobid, *_ = text.strip("\n").split(";")  # ignore cluster name

            return jobid
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError):
                e = subprocess.SubprocessError(e.stderr.strip("\n"))

            raise JobSubmissionError(str(job)) from e


class PBSScheduler(Scheduler):
    r"""PBS Pro scheduler.

    Jobs are submitted to the PBS queue via ``qsub``. Resources are allocated by
    the PBS manager according to the job and scheduler settings.  Job settings
    have precedence over scheduler settings.

    This implementation targets **PBS Pro** (as used on NASA Pleiades) which
    uses ``#PBS -J`` for array jobs and exposes the array index via the
    ``PBS_ARRAY_INDEX`` environment variable.

    Resource requests use the ``select`` statement so that chunked resources
    (``ncpus``, ``ngpus``, ``model``, ``mem``) are bundled together in a single
    ``#PBS -l select=…`` line, mirroring the style used in typical Pleiades
    submission scripts.

    A few convenience keys (e.g. ``cpus``, ``gpus``, ``ram``, ``timelimit``)
    are translated into their PBS equivalents.
    """

    backend: str = "pbs"

    # Keys that belong inside a ``select`` chunk
    SELECT_KEYS = {"ncpus", "ngpus", "model", "mem", "mpiprocs"}

    # Convenience translations (user-friendly name → PBS name)
    translate: Dict[str, str] = {
        "cpus": "ncpus",
        "gpus": "ngpus",
        "ram": "mem",
        "memory": "mem",
        "timelimit": "walltime",
    }

    def __init__(
        self,
        name: str = None,
        shell: str = "/bin/bash",
        interpreter: str = "python",
        env: Sequence[str] = [],  # noqa: B006
        **kwargs,
    ):
        r"""
        Arguments:
            name: The name of the workflow.
            shell: The scripting shell.
            interpreter: The Python interpreter.
            env: A sequence of commands to execute before each job is launched.
                 Use this for module loads, conda activations, environment
                 variable exports, etc.
            kwargs: Keyword arguments passed to :class:`Scheduler`.
        """

        super().__init__(name=name, **kwargs)

        assert shutil.which("qsub") is not None, "qsub executable not found"

        self.shell = shell
        self.interpreter = interpreter
        self.env = list(env)

    # ------------------------------------------------------------------
    # State inspection helpers
    # ------------------------------------------------------------------

    @lru_cache(None)  # noqa: B019
    def qstat(self, jobid: str) -> Dict[str, str]:
        r"""Return a mapping of ``key → value`` from ``qstat -f`` output."""
        try:
            text = subprocess.run(
                ["qstat", "-f", "-x", jobid],
                capture_output=True,
                check=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError:
            return {}

        result: Dict[str, str] = {}
        current_key = None
        current_val = ""

        for line in text.splitlines():
            stripped = line.strip()
            if " = " in stripped:
                if current_key is not None:
                    result[current_key] = current_val.strip()
                key, _, val = stripped.partition(" = ")
                current_key = key.strip()
                current_val = val.strip()
            elif current_key is not None and stripped:
                # continuation line
                current_val += stripped

        if current_key is not None:
            result[current_key] = current_val.strip()

        return result

    # Map PBS Pro states to human-readable strings
    _PBS_STATES: Dict[str, str] = {
        "Q": "QUEUED",
        "R": "RUNNING",
        "H": "HELD",
        "E": "EXITING",
        "F": "FINISHED",
        "S": "SUSPENDED",
        "W": "WAITING",
        "T": "TRANSITING",
        "B": "BEGUN",       # array job begun
        "X": "FINISHED",    # subjob finished
        "M": "MOVED",
    }

    def state(self, job: Job, i: int = None) -> str:
        if job in self.traces:
            return "CANCELLED"

        jobid = self.results[job]

        if job.array is not None and i is not None:
            # Query a specific sub-job
            jobid = f"{jobid}[{i}]"

        info = self.qstat(jobid)
        code = info.get("job_state", None)
        if code is None:
            return None
        return self._PBS_STATES.get(code, code)

    def output(self, job: Job, i: int = None) -> str:
        tag = self.tag(job)

        if job.array is None:
            logfile = self.path / f"{tag}.log"
        else:
            logfile = self.path / f"{tag}_{i}.log"

        if logfile.exists():
            with open(logfile, newline="", errors="replace") as f:
                return f.read()
        else:
            return None

    def report(self, job: Job = None) -> str:
        if job is None:
            headers = ("Name", "ID", "State")
            rows = []

            for job in self.order:
                if job in self.traces:
                    jobid = None
                else:
                    jobid = self.results[job]

                rows.append((str(job), jobid, self.state(job)))

            return tabulate(rows, headers, showindex=True)
        else:
            return super().report(job)

    def cancel(self, job: Job = None) -> str:
        if job is None:
            jobids = list(self.results.values())
        else:
            jobid = self.results[job]
            jobids = [jobid]

        return subprocess.run(
            ["qdel", *jobids],
            capture_output=True,
            check=True,
            text=True,
        ).stderr.strip("\n")

    # ------------------------------------------------------------------
    # Dependency satisfaction
    # ------------------------------------------------------------------

    async def satisfy(self, job: Job) -> str:
        results = await asyncio.gather(*map(self.submit, job.dependencies))

        for result in results:
            if isinstance(result, Exception):
                raise DependencyNeverSatisfiedError(str(job)) from result

    # ------------------------------------------------------------------
    # Job execution (submission)
    # ------------------------------------------------------------------

    async def exec(self, job: Job) -> Any:  # noqa: C901
        # ---- Build the submission script ----
        lines = [
            f"#!{self.shell}",
            "#",
            f"#PBS -N {job.name}",
        ]

        # Array jobs — PBS Pro uses ``-J``
        if job.array is not None:
            indices = comma_separated(job.array)

            if job.array_throttle is None:
                lines.append(f"#PBS -J {indices}")
            else:
                lines.append(f"#PBS -J {indices}%{job.array_throttle}")

        # Log files
        tag = self.tag(job)

        if job.array is None:
            logfile = self.path / f"{tag}.log"
        else:
            # PBS Pro substitutes the array index with ^array_index
            logfile = self.path / f"{tag}_^array_index.log"

        lines.append("#PBS -o /dev/null")
        lines.append("#PBS -e /dev/null")
        lines.append("#PBS -j oe")  # merge stdout and stderr

        # ---- Collect settings ----
        settings = self.settings.copy()
        settings.update(job.settings)

        # Translate convenience keys
        translated: Dict[str, Any] = {}
        for key, value in settings.items():
            key = self.translate.get(key, key)
            translated[key] = value
        settings = translated

        # Separate select-chunk keys from standalone -l keys and other flags
        select_parts: Dict[str, Any] = {}
        standalone_l: Dict[str, Any] = {}
        other_flags: Dict[str, str] = {}

        # ``nodes`` → ``select`` count (default 1)
        select_count = settings.pop("nodes", settings.pop("select", 1))

        for key, value in settings.items():
            if key in self.SELECT_KEYS:
                select_parts[key] = value
            elif key in ("walltime", "place", "filesystems"):
                standalone_l[key] = value
            elif key == "queue":
                other_flags["q"] = value
            elif key == "account":
                other_flags["A"] = value
            else:
                # Anything else goes into -l as a standalone resource
                standalone_l[key] = value

        # Build ``#PBS -l select=…`` line
        if select_parts:
            chunk = ":".join(f"{k}={v}" for k, v in select_parts.items())
            lines.append(f"#PBS -l select={select_count}:{chunk}")
        elif int(select_count) != 1:
            lines.append(f"#PBS -l select={select_count}")

        # Standalone ``-l`` resources (walltime, place, etc.)
        for key, value in standalone_l.items():
            lines.append(f"#PBS -l {key}={value}")

        # Other flags (queue, account, …)
        for flag, value in other_flags.items():
            lines.append(f"#PBS -{flag} {value}")

        # ---- Dependencies ----
        types = {
            "success": "afterok",
            "failure": "afternotok",
            "any": "afterany",
        }

        deps = []
        for dep, status in job.dependencies.items():
            depid = await self.submit(dep)
            deps.append(f"{types[status]}:{depid}")

        if deps:
            lines.append(f"#PBS -W depend={','.join(deps)}")

        lines.append("")

        # ---- Error handling ----
        lines.append("set -e")
        lines.append("")

        # ---- Redirect all output to log file in real-time ----
        # Using unbuffered redirection so output appears immediately
        lines.append(f"exec 1>{logfile} 2>&1")
        lines.append("")

        # ---- Environment setup ----
        if self.env:
            lines.extend([*self.env, ""])

        # ---- Pickle the job callable ----
        pklfile = self.path / f"{tag}.pkl"

        with open(pklfile, "wb") as f:
            pickle.dump(job.run, f)

        pyfile = self.path / f"{tag}.py"

        with open(pyfile, "w") as f:
            f.write(
                "\n".join([
                    "import argparse",
                    "import sys",
                    "import os",
                    "",
                    "# Ensure the working directory is on sys.path",
                    "# so that locally-pickled modules can be found",
                    "cwd = os.getcwd()",
                    "if cwd not in sys.path:",
                    "    sys.path.insert(0, cwd)",
                    "",
                    "import cloudpickle as pickle",
                    "",
                    "parser = argparse.ArgumentParser()",
                    "parser.add_argument('-i', '--index', type=int, default=None)",
                    "",
                    "args = parser.parse_args()",
                    "",
                    "with open('{}', 'rb') as f:".format(pklfile),
                    "    if args.index is None:",
                    "        pickle.load(f)()",
                    "    else:",
                    "        pickle.load(f)(args.index)",
                    "",
                ])
            )

        if job.interpreter is None:
            interpreter = self.interpreter
        else:
            interpreter = job.interpreter

        # PBS Pro exposes the array index as $PBS_ARRAY_INDEX
        # NOTE: Do NOT cd to $PBS_O_WORKDIR here — the env commands
        # already handle the working directory via the user-provided
        # cd command.  Adding cd $PBS_O_WORKDIR would undo that.
        if job.array is None:
            lines.append(f"{interpreter} {pyfile}")
        else:
            lines.append(f"{interpreter} {pyfile} -i $PBS_ARRAY_INDEX")

        lines.append("")

        # ---- Write the script ----
        shfile = self.path / f"{tag}.sh"

        with open(shfile, "w") as f:
            f.write("\n".join(lines))

        # ---- Submit ----
        try:
            text = subprocess.run(
                ["qsub", str(shfile)],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()

            # qsub returns a job id like "12345.pbspl4" — keep the full id
            jobid = text.split("\n")[-1].strip()

            return jobid
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError):
                e = subprocess.SubprocessError(e.stderr.strip("\n"))

            raise JobSubmissionError(str(job)) from e


class CyclicDependencyGraphError(Exception):
    pass


class DependencyNeverSatisfiedError(Exception):
    pass


class JobFailedError(Exception):
    pass


class JobNotFailedError(Exception):
    pass


class JobSubmissionError(Exception):
    pass
