// Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.
//
// Replay in Pyodide (WebAssembly, the runtime of DataLab-Web) the calls of
// Sigima's validation tests captured in CPython, as an application replays a
// recorded activity, and save the results for a comparison in CPython
// (sigima.tests.replay_probe.compare_replays).
//
// Usage:
//   node run.mjs --sigima <sigima.whl> --guidata <guidata.whl>
//                --capture <dir> --results <dir> [--report <file.json>]
//                [--batch <operations per Pyodide instance>] [--operations a,b]
//
// <dir> of --capture is written by the test suite run with
// SIGIMA_REPLAY_DUMP=<dir>/ (one file per operation), with the NumPy version of
// Pyodide (1.26) so that the captured objects unpickle here. Operations are
// replayed in batches, each in a fresh Pyodide instance (the WebAssembly heap
// never shrinks); an operation whose batch fails is retried alone, and reported
// as an error if it still fails. Packages of the Pyodide distribution are
// fetched from the Pyodide CDN on first use; other dependencies from PyPI.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { loadPyodide } from "pyodide";

function parseArgs(argv) {
  const args = { batch: "25" };
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith("--") || argv[i + 1] === undefined) {
      throw new Error(`Invalid argument: ${argv[i]}`);
    }
    args[argv[i].slice(2)] = argv[i + 1];
  }
  for (const name of ["sigima", "guidata", "capture", "results"]) {
    if (!args[name]) {
      throw new Error(`Missing --${name} (see the header of run.mjs)`);
    }
  }
  if (!/^[1-9]\d*$/.test(args.batch)) {
    throw new Error(`Invalid --batch ${args.batch}: a positive integer is expected`);
  }
  return args;
}

async function startPyodide(wheels) {
  const py = await loadPyodide();
  await py.loadPackage(["numpy", "scipy", "pandas", "pillow", "micropip"]);
  const specs = [];
  for (const wheel of wheels) {
    const name = basename(wheel);
    py.FS.writeFile(`/tmp/${name}`, readFileSync(wheel));
    specs.push(`emfs:/tmp/${name}`);
  }
  // Same requirements as DataLab-Web (tifffile capped for NumPy 1.26), plus
  // OpenCV, used by one blob detection.
  specs.push("tifffile<2025", "opencv-python");
  await py.runPythonAsync(`
import os
os.environ["LANG"] = os.environ["LANGUAGE"] = "en"
import micropip
await micropip.install(${JSON.stringify(specs)})
`);
  return py;
}

async function replayBatch(args, operations) {
  const py = await startPyodide([resolve(args.sigima), resolve(args.guidata)]);
  for (const [mount, dir] of [
    ["/capture", args.capture],
    ["/results", args.results],
  ]) {
    py.FS.mkdirTree(mount);
    py.FS.mount(py.FS.filesystems.NODEFS, { root: resolve(dir) }, mount);
  }
  const output = await py.runPythonAsync(`
import json, sys
import numpy, scipy, skimage, sigima
from sigima.tests import replay_probe
json.dumps({
    "environment": {
        "python": sys.version.split()[0], "numpy": numpy.__version__,
        "scipy": scipy.__version__, "scikit-image": skimage.__version__,
        "sigima": sigima.__version__,
    },
    "operations": replay_probe.replay_dump(
        "/capture/", "/results", operations=${JSON.stringify(operations)}
    ),
})
`);
  return JSON.parse(output);
}

const args = parseArgs(process.argv.slice(2));
mkdirSync(args.results, { recursive: true });
let operations = readdirSync(args.capture)
  .filter((name) => name.endsWith(".pkl"))
  .map((name) => name.slice(0, -".pkl".length))
  .sort();
if (args.operations) {
  const wanted = new Set(args.operations.split(","));
  operations = operations.filter((op) => wanted.has(op));
}
const size = Number(args.batch);
const report = { environment: null, operations: {} };
const merge = (output) => {
  report.environment = output.environment;
  Object.assign(report.operations, output.operations);
};
for (let start = 0; start < operations.length; start += size) {
  const batch = operations.slice(start, start + size);
  console.log(`Replaying ${start + 1}-${start + batch.length} of ${operations.length}`);
  try {
    merge(await replayBatch(args, batch));
  } catch (error) {
    console.warn(`Batch failed (${String(error.message).split("\n")[0]}), retrying alone`);
    for (const op of batch) {
      try {
        merge(await replayBatch(args, [op]));
      } catch (single) {
        const lines = String(single.message).trim().split("\n");
        report.operations[op] = { error: lines.slice(-2).join(" ") };
        console.warn(`${op}: ${report.operations[op].error}`);
      }
    }
  }
}
console.log(JSON.stringify(report.environment));
const verdicts = {};
let replayErrors = 0;
for (const entry of Object.values(report.operations)) {
  // Captured calls whose replay raised (replay_probe status "error")
  const failed = entry.statuses?.error ?? 0;
  replayErrors += failed;
  let key = "not_replayed";
  if (entry.error) {
    key = "error";
  } else if (entry.samples) {
    key = entry.verdict;
  } else if (failed) {
    key = "error";
  }
  verdicts[key] = (verdicts[key] ?? 0) + 1;
}
console.log(`Verdicts against the captured results: ${JSON.stringify(verdicts)}`);
if (replayErrors) {
  console.log(`Captured calls failing to replay: ${replayErrors}`);
}
if (args.report) {
  writeFileSync(args.report, JSON.stringify(report, null, 1));
}
if (verdicts.error || replayErrors) {
  process.exitCode = 1;
}
