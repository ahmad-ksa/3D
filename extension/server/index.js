// Studio3D extension launcher for the Claude desktop app.
// Runs with Claude's built-in Node.js and starts the Studio3D MCP server using the
// private Python that install.ps1 put in %LOCALAPPDATA%\Studio3D (no system Python needed).
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const dir = process.env.STUDIO3D_DIR ||
  path.join(process.env.LOCALAPPDATA || path.join(require('os').homedir(), 'AppData', 'Local'), 'Studio3D');
const python = path.join(dir, 'python', 'tools', process.platform === 'win32' ? 'python.exe' : 'python');
const app = path.join(dir, 'app');

if (!fs.existsSync(python) || !fs.existsSync(path.join(app, 'studio3d'))) {
  process.stderr.write(`Studio3D is not installed in ${dir}. Install it first: ` +
    'irm https://raw.githubusercontent.com/ahmad-ksa/3D/main/install.ps1 | iex\n');
  process.exit(1);
}

const child = spawn(python, ['-m', 'studio3d.mcp_server'], {
  cwd: app,
  stdio: 'inherit',
  windowsHide: true,
  env: { ...process.env, PYTHONPATH: app, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
});
child.on('exit', (code) => process.exit(code ?? 0));
child.on('error', (err) => { process.stderr.write(String(err) + '\n'); process.exit(1); });
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => child.kill(sig));
