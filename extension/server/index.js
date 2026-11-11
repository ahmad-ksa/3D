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
  cwd: app,
  stdio: ['pipe', 'pipe', 'pipe'],
  windowsHide: true,
  env: { ...process.env, PYTHONPATH: app, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
});
// relay the MCP stdio channel explicitly (works with any parent transport)
process.stdin.pipe(child.stdin);
child.stdout.pipe(process.stdout);
child.stderr.on('data', (d) => { process.stderr.write(d); try { fs.appendFileSync(path.join(dir, 'extension.log'), d); } catch (e) {} });
child.on('exit', (code, sig) => { log(`python exited code=${code} signal=${sig}`); process.exit(code ?? 0); });
child.on('error', (err) => { log(`spawn error: ${err}`); process.exit(1); });
process.stdin.on('end', () => child.stdin.end());
for (const s of ['SIGINT', 'SIGTERM']) process.on(s, () => child.kill(s));
