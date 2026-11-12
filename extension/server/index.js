// Studio3D extension launcher for the Claude desktop app.
// Runs with Claude's built-in Node.js and starts the Studio3D MCP server using the
// private Python that install.ps1 put in %LOCALAPPDATA%\Studio3D (no system Python needed).
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const exe = process.platform === 'win32' ? 'python.exe' : 'python';
const candidates = [
  process.env.STUDIO3D_DIR,
  process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Studio3D'),
  path.join(os.homedir(), 'AppData', 'Local', 'Studio3D'),
].filter((d) => d && !d.includes('${'));

const dir = candidates.find((d) => fs.existsSync(path.join(d, 'python', 'tools', exe)) &&
                                   fs.existsSync(path.join(d, 'app', 'studio3d')));

function log(msg) {
  const line = `${new Date().toISOString()} ${msg}\n`;
  process.stderr.write(line);
  for (const d of [dir, ...candidates]) {
    try { if (d && fs.existsSync(d)) { fs.appendFileSync(path.join(d, 'extension.log'), line); return; } } catch (e) {}
  }
}

if (!dir) {
  log(`Studio3D not found. Tried: ${JSON.stringify(candidates)} (STUDIO3D_DIR=${process.env.STUDIO3D_DIR})`);
  process.exit(1);
}

const python = path.join(dir, 'python', 'tools', exe);
const app = path.join(dir, 'app');
log(`starting ${python} -m studio3d.mcp_server (cwd ${app})`);

const child = spawn(python, ['-m', 'studio3d.mcp_server'], {
  cwd: dir,  // not the app folder, so updates can replace it
  stdio: ['pipe', 'pipe', 'pipe'],
  windowsHide: true,
  env: { ...process.env, PYTHONPATH: app, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
});
// Relay the MCP stdio channel line by line. Newer Claude clients first send requests that the
// server's MCP library (1.6) does not know (e.g. "server/discover") and that would crash it:
// answer those here with "method not found" so the client falls back to the classic handshake.
const KNOWN = /^(initialize|ping|tools\/|resources\/|prompts\/|logging\/|completion\/|notifications\/)/;
let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, i);
    buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let msg = null;
    try { msg = JSON.parse(line); } catch (e) {}
    if (msg && typeof msg.method === 'string' && !KNOWN.test(msg.method)) {
      if (msg.id !== undefined && msg.id !== null) {
        process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: msg.id,
          error: { code: -32601, message: `Method not found: ${msg.method}` } }) + '\n');
      }
      continue;  // unknown notification: drop
    }
    child.stdin.write(line + '\n');
  }
});
child.stdout.pipe(process.stdout);
child.stderr.on('data', (d) => { process.stderr.write(d); try { fs.appendFileSync(path.join(dir, 'extension.log'), d); } catch (e) {} });
child.on('exit', (code, sig) => { log(`python exited code=${code} signal=${sig}`); process.exit(code ?? 0); });
child.on('error', (err) => { log(`spawn error: ${err}`); process.exit(1); });
process.stdin.on('end', () => { if (buf.trim()) child.stdin.write(buf + '\n'); child.stdin.end(); });
for (const s of ['SIGINT', 'SIGTERM']) process.on(s, () => child.kill(s));
