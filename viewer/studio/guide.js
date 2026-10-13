// The picture guide (docs/prompt-guide.md) as a pop-up over the studio. It reads the same file
// GitHub shows, so the guide is written once. The Markdown it needs is small: headings,
// paragraphs, bold, links, lists, one image and a code block. No DOM in the renderer, so
// Node tests import this exact file.

export const GUIDE_URL = '/docs/prompt-guide.md';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function inline(text, base) {
  return esc(text)
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) => `<img src="${new URL(src, base).pathname}" alt="${alt}">`)
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => `<a href="${new URL(href, base).href}" target="_blank" rel="noopener">${label}</a>`)
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

/** The guide's Markdown as HTML. `base` is the file's own URL, so its relative image and
 *  link paths resolve the way they do on GitHub. */
export function renderGuide(markdown, base = `http://studio${GUIDE_URL}`) {
  const out = [];
  let paragraph = [], list = null, code = null;
  const flush = () => {
    if (paragraph.length) out.push(`<p>${inline(paragraph.join(' '), base)}</p>`);
    if (list) out.push(`<ul>${list.map((item) => `<li>${inline(item, base)}</li>`).join('')}</ul>`);
    paragraph = []; list = null;
  };
  for (const line of String(markdown).replace(/\r\n/g, '\n').split('\n')) {
    if (code) {
      if (line.startsWith('```')) { out.push(`<pre class="log">${esc(code.join('\n'))}</pre>`); code = null; }
      else code.push(line);
      continue;
    }
    if (line.startsWith('```')) { flush(); code = []; continue; }
    const heading = line.match(/^(#{1,3})\s+(.*)/);
    if (heading) { flush(); out.push(`<h${heading[1].length + 1}>${inline(heading[2], base)}</h${heading[1].length + 1}>`); continue; }
    const item = line.match(/^\s*-\s+(.*)/);
    if (item) { if (paragraph.length) flush(); (list ??= []).push(item[1]); continue; }
    if (!line.trim()) { flush(); continue; }
    if (list) list[list.length - 1] += ` ${line.trim()}`;   // a list item wrapped onto the next line
    else paragraph.push(line.trim());
  }
  if (code) out.push(`<pre class="log">${esc(code.join('\n'))}</pre>`);
  flush();
  return out.join('\n');
}

/** Whether a clicked link points at the picture guide (on GitHub or here). */
export function isGuideLink(href) {
  return /\/prompt-guide\.md(?:[#?].*)?$/.test(href || '');
}

/** Open the guide on top of everything, Create included, so nothing typed there is lost. */
export async function openPromptGuide() {
  document.querySelector('.modal')?.remove();
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.innerHTML = `<div class="modal-card sheet-card guide-doc" role="dialog" aria-modal="true" aria-label="Picture guide">
    <div class="sheet-top"><span class="meta">Picture guide</span><button type="button" class="ghost" data-close>Close</button></div>
    <div class="guide-body"><p class="meta">Loading…</p></div></div>`;
  document.body.appendChild(modal);
  const close = () => { modal.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
  modal.querySelector('[data-close]').onclick = close;
  modal.querySelector('[data-close]').focus();
  const body = modal.querySelector('.guide-body');
  try {
    const response = await fetch(GUIDE_URL);
    if (!response.ok) throw new Error(response.statusText);
    body.innerHTML = renderGuide(await response.text(), new URL(GUIDE_URL, location.href).href);
  } catch {
    body.innerHTML = '<p class="meta">The guide did not load. It is also on <a href="https://github.com/Bingeljell/image-to-3dlab/blob/main/docs/prompt-guide.md" target="_blank" rel="noopener">GitHub</a>.</p>';
  }
}
