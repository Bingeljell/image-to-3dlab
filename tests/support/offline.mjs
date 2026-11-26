// Guard standard Node APIs in the fixture children preloading this module.
import childProcess from 'node:child_process';
import dgram from 'node:dgram';
import dns from 'node:dns';
import net from 'node:net';
import { syncBuiltinESMExports } from 'node:module';

function blocked(operation) {
  throw new Error(`Offline tests block ${operation}`);
}

function checkHost(host) {
  if (host == null || ['localhost', 'localhost.', '::1'].includes(host) ||
      (net.isIPv4(host) && host.startsWith('127.'))) return;
  blocked(`connection or lookup for ${host}`);
}

function guardLookup(lookup) {
  return function (host, ...args) {
    checkHost(host);
    return lookup.call(this, host, ...args);
  };
}

dns.lookup = guardLookup(dns.lookup);
const lookup = dns.promises.lookup;
dns.promises.lookup = async (...args) => guardLookup(lookup).apply(dns.promises, args);
dns.lookupService = () => blocked('reverse name lookup');
dns.promises.lookupService = async () => blocked('reverse name lookup');
for (const api of [dns, dns.promises, dns.Resolver.prototype, dns.promises.Resolver.prototype]) {
  for (const name of Object.keys(api)) {
    if (/^(resolve|reverse)/.test(name)) api[name] = () => blocked('direct DNS queries');
  }
}

const connect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function (...args) {
  const first = Array.isArray(args[0]) ? args[0][0] : args[0];
  const options = typeof first === 'object' ? first : { port: first, host: args[1] };
  const unixPath = options.path || (typeof first === 'string' && !/^\d+$/.test(first));
  if (!unixPath) {
    checkHost(typeof options.host === 'function' ? undefined : options.host);
    if (Number(options.port) === 9876) blocked('connection to Blender');
  }
  return connect.apply(this, args);
};

dgram.createSocket = () => blocked('UDP sockets');
for (const name of ['exec', 'execFile', 'execFileSync', 'execSync', 'fork', 'spawn', 'spawnSync']) {
  childProcess[name] = () => blocked('child processes from Node fixtures');
}
childProcess.ChildProcess.prototype.spawn = () => blocked('child processes from Node fixtures');
syncBuiltinESMExports();
