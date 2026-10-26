// Marks the studio's own writes. The server refuses a POST without this header, and a
// form on another site cannot send one, so a web page cannot delete weights or start jobs
// here behind your back.
(() => {
  const send = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const request = input instanceof Request ? input : null;
    const url = new URL(request ? request.url : String(input), location.href);
    const method = String(init.method || (request ? request.method : 'GET')).toUpperCase();
    if (url.origin !== location.origin || method === 'GET' || method === 'HEAD') return send(input, init);
    const headers = new Headers(init.headers || (request ? request.headers : undefined));
    if (!headers.has('X-AssetFurnace-Request')) headers.set('X-AssetFurnace-Request', 'studio');
    return send(input, { ...init, headers });
  };
})();
