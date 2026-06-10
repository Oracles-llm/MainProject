const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");

let backendProcess = null;
let mainWindow = null;

const PACKAGED_LLM_MODEL_FILE = "qwen2.5-3b-instruct-q4_0.gguf";
const PACKAGED_EMBEDDING_MODEL_FILE = "qwen3-embedding-0.6b-q8_0.gguf";

function projectRoot() {
  return path.resolve(__dirname, "..", "..");
}

function findAvailablePort(startPort) {
  return new Promise((resolve, reject) => {
    let port = startPort;

    const tryPort = () => {
      const server = net.createServer();
      server.once("error", () => {
        port += 1;
        if (port > startPort + 99) {
          reject(new Error(`No available port found from ${startPort} to ${startPort + 99}.`));
          return;
        }
        tryPort();
      });
      server.once("listening", () => {
        server.close(() => resolve(port));
      });
      server.listen(port, "127.0.0.1");
    };

    tryPort();
  });
}

function parseDotEnv(filePath) {
  if (!fs.existsSync(filePath)) return {};

  const result = {};
  const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const equalsIndex = trimmed.indexOf("=");
    if (equalsIndex === -1) continue;

    const key = trimmed.slice(0, equalsIndex).trim();
    let value = trimmed.slice(equalsIndex + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    result[key] = value;
  }

  return result;
}

function packagedResourcePath(...segments) {
  if (!app.isPackaged) return null;

  const resourcePath = path.join(process.resourcesPath, ...segments);
  return fs.existsSync(resourcePath) ? resourcePath : null;
}

function packagedDefaultEnvironment() {
  const llmModelPath = packagedResourcePath("models", PACKAGED_LLM_MODEL_FILE);
  const embeddingModelPath = packagedResourcePath("models", PACKAGED_EMBEDDING_MODEL_FILE);

  return {
    LLM_PROVIDER: "llama_cpp",
    QDRANT_MODE: "local",
    EMBEDDING_PROVIDER: "fastembed",
    ...(llmModelPath ? { LLM_MODEL_PATH: llmModelPath } : {}),
    ...(embeddingModelPath ? { EMBEDDING_MODEL_PATH: embeddingModelPath } : {}),
  };
}

function backendEnvironment(apiPort) {
  const envFiles = app.isPackaged
    ? [path.join(process.resourcesPath, ".env"), path.join(app.getPath("userData"), ".env")]
    : [path.join(projectRoot(), ".env")];
  const fileEnv = Object.assign({}, ...envFiles.map(parseDotEnv));

  return {
    ...process.env,
    ...packagedDefaultEnvironment(),
    ...fileEnv,
    LOG_DIR: fileEnv.LOG_DIR || path.join(app.getPath("userData"), "logs"),
    QDRANT_LOCAL_PATH: fileEnv.QDRANT_LOCAL_PATH || path.join(app.getPath("userData"), "qdrant"),
    PORT: String(apiPort),
  };
}

function backendCommand(apiPort) {
  if (app.isPackaged) {
    const exeName = process.platform === "win32" ? "oracles-backend.exe" : "oracles-backend";
    return {
      command: path.join(process.resourcesPath, "backend", exeName),
      args: ["--host", "127.0.0.1", "--port", String(apiPort)],
      cwd: process.resourcesPath,
    };
  }

  return {
    command: process.platform === "win32" ? "python" : "python3",
    args: ["main.py", "--host", "127.0.0.1", "--port", String(apiPort)],
    cwd: projectRoot(),
  };
}

function startBackend(apiPort) {
  const backend = backendCommand(apiPort);

  backendProcess = spawn(backend.command, backend.args, {
    cwd: backend.cwd,
    env: backendEnvironment(apiPort),
    stdio: app.isPackaged ? "ignore" : "inherit",
    windowsHide: true,
  });

  backendProcess.once("exit", (code) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("backend-exited", code);
    }
  });
}

function stopBackend() {
  if (!backendProcess || backendProcess.killed) return;

  backendProcess.kill();
  backendProcess = null;
}

async function waitForBackend(apiBaseUrl) {
  const deadline = Date.now() + 180_000;
  const healthUrl = `${apiBaseUrl}/api/v1/health`;
  let lastError = "backend did not become ready";

  while (Date.now() < deadline) {
    try {
      const response = await fetch(healthUrl);
      if (response.ok) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error.message;
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  throw new Error(`Timed out waiting for ${healthUrl}: ${lastError}`);
}

async function createWindow(apiBaseUrl) {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 760,
    minWidth: 900,
    minHeight: 620,
    title: "Oracles AI Chat Companion",
    backgroundColor: "#ffffff",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  const indexPath = path.join(__dirname, "..", "dist", "index.html");
  await mainWindow.loadFile(indexPath, {
    query: {
      apiBaseUrl,
    },
  });
}

async function boot() {
  try {
    const apiPort = await findAvailablePort(8000);
    const apiBaseUrl = `http://127.0.0.1:${apiPort}`;

    startBackend(apiPort);
    await waitForBackend(apiBaseUrl);
    await createWindow(apiBaseUrl);
  } catch (error) {
    dialog.showErrorBox("Startup failed", error instanceof Error ? error.message : String(error));
    app.quit();
  }
}

app.whenReady().then(boot);

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", stopBackend);

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    boot();
  }
});
