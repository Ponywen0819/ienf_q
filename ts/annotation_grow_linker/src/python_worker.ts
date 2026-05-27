/**
 * PythonWorker — spawns `tools/electron_worker.py` and exchanges line-buffered
 * JSON-RPC frames over stdin/stdout. One request in flight at a time per call;
 * concurrent `call()`s are multiplexed by id.
 *
 * Stdout: protocol only. Stderr: forwarded to our stderr so Python logs and
 * tracebacks remain visible during development.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createInterface, type Interface as ReadlineInterface } from "node:readline";
import { EOL } from "node:os";

export interface RpcErrorBody {
  type: string;
  message: string;
  traceback?: string;
}

export class PythonRpcError extends Error {
  constructor(public readonly remote: RpcErrorBody) {
    super(`${remote.type}: ${remote.message}`);
    this.name = "PythonRpcError";
  }
}

export interface PythonWorkerOptions {
  /** Default: "uv". Override to e.g. an absolute path or a venv python. */
  command?: string;
  /** Default: ["run", "python", "tools/electron_worker.py"]. */
  args?: readonly string[];
  /** Default: process.cwd(). For tests, point at the ienf_q repo root. */
  cwd?: string;
  /** Extra env vars; merged onto process.env. */
  env?: Readonly<Record<string, string>>;
  /** If true, swallow worker stderr instead of piping to ours. */
  silenceStderr?: boolean;
}

interface Pending {
  resolve(value: unknown): void;
  reject(err: Error): void;
}

export class PythonWorker {
  private readonly proc: ChildProcess;
  private readonly rl: ReadlineInterface;
  private readonly pending = new Map<string, Pending>();
  private nextId = 0;
  private exited = false;
  private exitError: Error | null = null;

  constructor(options: PythonWorkerOptions = {}) {
    const cmd = options.command ?? "uv";
    const args = options.args ?? ["run", "python", "tools/electron_worker.py"];
    this.proc = spawn(cmd, [...args], {
      cwd: options.cwd,
      env: { ...process.env, ...options.env },
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.rl = createInterface({ input: this.proc.stdout! });
    this.rl.on("line", (line) => this.handleLine(line));

    if (!options.silenceStderr) {
      this.proc.stderr!.on("data", (chunk: Buffer) => {
        process.stderr.write(chunk);
      });
    }

    this.proc.on("error", (err) => {
      this.exited = true;
      this.exitError = err;
      this.rejectAllPending(err);
    });

    this.proc.on("exit", (code, signal) => {
      this.exited = true;
      const reason = code != null ? `code ${code}` : `signal ${signal}`;
      this.exitError = new Error(`python worker exited (${reason})`);
      this.rejectAllPending(this.exitError);
    });
  }

  private rejectAllPending(err: Error): void {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  private handleLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return;
    let frame: { id?: unknown; result?: unknown; error?: RpcErrorBody };
    try {
      frame = JSON.parse(trimmed);
    } catch {
      process.stderr.write(
        `[python-worker] non-JSON on stdout: ${trimmed}${EOL}`,
      );
      return;
    }
    if (frame.id == null) {
      process.stderr.write(
        `[python-worker] frame without id: ${trimmed}${EOL}`,
      );
      return;
    }
    const id = String(frame.id);
    const p = this.pending.get(id);
    if (!p) {
      process.stderr.write(`[python-worker] unknown id ${id}${EOL}`);
      return;
    }
    this.pending.delete(id);
    if (frame.error) {
      p.reject(new PythonRpcError(frame.error));
    } else {
      p.resolve(frame.result);
    }
  }

  /** Send one RPC. Resolves with `result`; rejects on `error` or worker death. */
  call<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<T> {
    if (this.exited) {
      return Promise.reject(
        this.exitError ?? new Error("python worker has exited"),
      );
    }
    const id = String(this.nextId++);
    const payload = JSON.stringify({ id, method, params }) + "\n";
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.proc.stdin!.write(payload, (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  /** Quick liveness probe; throws if the worker isn't responding. */
  async ready(): Promise<void> {
    const r = await this.call<string>("ping");
    if (r !== "pong") {
      throw new Error(`ping returned ${JSON.stringify(r)}`);
    }
  }

  /** Close stdin and wait for the worker to exit. */
  async close(): Promise<void> {
    if (this.exited) return;
    this.proc.stdin!.end();
    await new Promise<void>((resolve) => {
      this.proc.once("exit", () => resolve());
      setTimeout(() => {
        if (!this.exited) this.proc.kill();
        resolve();
      }, 2000).unref();
    });
  }
}
