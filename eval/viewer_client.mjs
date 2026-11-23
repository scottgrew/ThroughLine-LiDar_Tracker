#!/usr/bin/env node
"use strict";

import { writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const TOPICS = new Set(["/points", "/tf", "/scene"]);

function percentile(sorted, fraction) {
  if (sorted.length === 0) return 0;
  const index = Math.floor(fraction * (sorted.length - 1));
  return sorted[index];
}

const { values } = parseArgs({
  options: {
    port: { type: "string" },
    out: { type: "string" },
  },
});

const port = values.port;
const outPath = values.out;
if (!port || !outPath) {
  console.error("usage: node eval/viewer_client.mjs --port N --out PATH");
  process.exit(1);
}

const messages = { "/points": 0, "/tf": 0, "/scene": 0 };
const bytes = { "/points": 0, "/tf": 0, "/scene": 0 };
const topicBySubId = new Map();
let subscribed = 0;
let sentSubscribe = false;
const sceneArrivals = [];
const sceneLogTimes = [];
let failed = false;

function writeResult(closedByServer) {
  const jitter = sceneArrivals.map((arrival, index) => {
    const logDelta = (sceneLogTimes[index] - sceneLogTimes[0]) / 1e6;
    return arrival - sceneArrivals[0] - logDelta;
  });
  const sorted = [...jitter].sort((left, right) => left - right);
  const ok = closedByServer && subscribed === 3 && sentSubscribe && !failed;
  const payload = {
    subscribed,
    messages,
    bytes,
    jitter_ms: {
      p50: percentile(sorted, 0.5),
      p99: percentile(sorted, 0.99),
      max: sorted.length ? sorted[sorted.length - 1] : 0,
    },
    closed_by_server: ok,
  };
  writeFileSync(outPath, `${JSON.stringify(payload)}\n`, "utf8");
}

function finish(code, closedByServer) {
  writeResult(closedByServer);
  process.exit(code);
}

function onAdvertise(text) {
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    return;
  }
  if (payload.op !== "advertise" || !Array.isArray(payload.channels)) return;
  const subs = [];
  let nextId = 1;
  for (const channel of payload.channels) {
    if (!TOPICS.has(channel.topic)) continue;
    topicBySubId.set(nextId, channel.topic);
    subs.push({ id: nextId, channelId: channel.id });
    nextId += 1;
  }
  if (subs.length !== 3) return;
  ws.send(JSON.stringify({ op: "subscribe", subscriptions: subs }));
  sentSubscribe = true;
  subscribed = 3;
}

function onBinary(data) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (view.byteLength < 13 || view.getUint8(0) !== 0x01) return;
  const subId = view.getUint32(1, true);
  const logTime = view.getBigUint64(5, true);
  const topic = topicBySubId.get(subId);
  if (!topic) return;
  const payloadBytes = view.byteLength - 13;
  messages[topic] += 1;
  bytes[topic] += payloadBytes;
  if (topic === "/scene") {
    sceneArrivals.push(performance.now());
    sceneLogTimes.push(Number(logTime));
  }
}

let ws;
const deadline = Date.now() + 15000;

function tryConnect() {
  if (Date.now() >= deadline) finish(1, false);
  ws = new WebSocket(`ws://127.0.0.1:${port}`, "foxglove.sdk.v1");
  ws.binaryType = "arraybuffer";
  ws.addEventListener("message", (event) => {
    if (typeof event.data === "string") onAdvertise(event.data);
    else onBinary(new Uint8Array(event.data));
  });
  ws.addEventListener("error", () => {
    ws.close();
    setTimeout(tryConnect, 200);
  });
  ws.addEventListener("close", () => {
    if (!sentSubscribe) return;
    finish(sentSubscribe && subscribed === 3 && !failed ? 0 : 1, true);
  });
}

tryConnect();