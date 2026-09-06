import { expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { saveCache } from "../src/cache.ts";
import type { VerifierClient } from "../src/client.ts";
import type { Trial } from "../src/loader.ts";
import { cacheContext } from "../src/run.ts";

const cacheModule = new URL("../src/cache.ts", import.meta.url).href;

test("cache identity follows mutations to an existing image array", () => {
  const images = [{ type: "image" as const, mimeType: "image/png", data: "before" }];
  const trials: Trial[] = [0, 1].map((i) => ({
    trialName: String(i), reward: 0, problem: "task", trace: String(i), images,
  }));
  const context = () => cacheContext({} as VerifierClient, trials, 0, 1, 0, "", {
    id: "criterion", name: "Criterion", description: "Evaluate evidence",
  });
  const before = context().imagesFingerprint;
  images[0].data = "after";
  expect(context().imagesFingerprint).not.toBe(before);
  const after = context().imagesFingerprint;
  images.push({ type: "image", mimeType: "image/png", data: "another" });
  expect(context().imagesFingerprint).not.toBe(after);
});

test("independent processes merge concurrent cache writes without losing keys", async () => {
  const directory = mkdtempSync(join(tmpdir(), "verifier-concurrent-"));
  const cacheFile = join(directory, "cache.json");
  const start = Date.now() + 300;
  const children = Array.from({ length: 6 }, (_, i) => Bun.spawn([
    process.execPath, "-e", `
      import {saveCache} from ${JSON.stringify(cacheModule)};
      await Bun.sleep(Math.max(0, ${start} - Date.now()));
      saveCache(${JSON.stringify(cacheFile)}, {writer_${i}: {score_A: 0.75, score_B: 0.25}});
    `,
  ], { stdout: "pipe", stderr: "pipe" }));
  try {
    expect(await Promise.all(children.map((child) => child.exited))).toEqual(Array(6).fill(0));
    expect(Object.keys(JSON.parse(readFileSync(cacheFile, "utf8"))).sort()).toEqual(
      Array.from({ length: 6 }, (_, i) => `writer_${i}`),
    );

  } finally {
    for (const child of children) child.kill();
    rmSync(directory, { recursive: true, force: true });
  }
});

test("legacy lock directories are preserved even when their owner is stale", () => {
  const directory = mkdtempSync(join(tmpdir(), "verifier-legacy-lock-"));
  const cacheFile = join(directory, "cache.json");
  const lock = `${cacheFile}.lock`;
  mkdirSync(lock);
  const metadata = JSON.stringify({ pid: 2147483647, token: "legacy", createdAt: 0 });
  writeFileSync(join(lock, "owner"), metadata);
  try {
    expect(() => saveCache(cacheFile, { ours: { score_A: 0.5, score_B: 0.5 } })).toThrow(
      "Legacy cache lock directory",
    );
    expect(readFileSync(join(lock, "owner"), "utf8")).toBe(metadata);
    expect(existsSync(cacheFile)).toBe(false);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("OS cache locks exclude live owners and recover after abrupt process death", async () => {
  const directory = mkdtempSync(join(tmpdir(), "verifier-os-lock-"));
  const cacheFile = join(directory, "cache.json");
  // Canonicalize exactly as the writer does; macOS /tmp is a symlink.
  const { realpathSync } = await import("node:fs");
  const lockPath = join(realpathSync(directory), "cache.json.lock");
  const owner = Bun.spawn([process.execPath, "-e", `
    import {FileLock} from "@oh-my-pi/pi-natives";
    const lock = FileLock.tryAcquire(${JSON.stringify(lockPath)});
    if (!lock.acquired) throw new Error("owner could not acquire lock");
    console.log("ready");
    await Bun.sleep(60000);
    lock.release();
  `], { stdout: "pipe", stderr: "pipe" });
  try {
    const reader = owner.stdout.getReader();
    const ready = await reader.read();
    expect(new TextDecoder().decode(ready.value)).toContain("ready");
    reader.releaseLock();
    const contender = Bun.spawn([process.execPath, "-e", `
      import {FileLock} from "@oh-my-pi/pi-natives";
      const lock = FileLock.tryAcquire(${JSON.stringify(lockPath)});
      if (lock.acquired) throw new Error("acquired a live owner's lock");
    `], { stdout: "pipe", stderr: "pipe" });
    expect(await contender.exited).toBe(0);
    owner.kill("SIGKILL");
    await owner.exited;
    saveCache(cacheFile, { recovered: { score_A: 0.75, score_B: 0.25 } });
    expect(JSON.parse(readFileSync(cacheFile, "utf8"))).toHaveProperty("recovered");
  } finally {
    owner.kill();
    await owner.exited;
    rmSync(directory, { recursive: true, force: true });
  }
});
