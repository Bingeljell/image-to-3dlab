/** Show or hide the Library controls while its first request is pending. */
export function setLibraryLoading(root, loading) {
  const controls = root.querySelector('#libControls');
  const indicator = root.querySelector('#libLoad');
  if (controls) {
    controls.hidden = loading;
    controls.inert = loading;
  }
  if (indicator) {
    indicator.hidden = !loading;
    if (loading) {
      indicator.textContent = '';
      const doc = root.ownerDocument || document;
      const spinner = doc.createElement('span');
      spinner.className = 'loader';
      spinner.setAttribute('aria-hidden', 'true');
      const text = doc.createElement('span');
      text.textContent = 'Loading your assets…';
      indicator.append(spinner, text);
    }
  }
}

/** Let only the newest Library request commit its result or error. */
export async function runLibraryLoad({ requestId, isCurrent, fetchAssets, onSuccess, onError }) {
  try {
    const payload = await fetchAssets();
    if (!isCurrent()) return false;
    onSuccess(payload);
  } catch (error) {
    if (!isCurrent()) return false;
    onError(error);
  }
  return true;
}

/** Keep the initial controls hidden while offering an actionable load failure. */
export function showLibraryLoadError(root, message, retry) {
  const indicator = root.querySelector('#libLoad');
  if (!indicator) return;
  indicator.hidden = false;
  indicator.textContent = '';
  const doc = root.ownerDocument || document;
  const text = doc.createElement('span');
  text.textContent = `The Library didn't load: ${message}`;
  const button = doc.createElement('button');
  button.className = 'linkish';
  button.type = 'button';
  button.textContent = 'Try again';
  button.onclick = () => {
    button.disabled = true;
    retry();
  };
  indicator.append(text, button);
}

/** Replace the footer status with an actionable error, without inserting server text as markup. */
export function showLibraryError(root, message, retry) {
  const footer = root.querySelector('#libFoot');
  if (!footer) return;
  footer.textContent = '';
  const doc = root.ownerDocument || document;
  const text = doc.createElement('span');
  text.textContent = `The Library didn't load: ${message}`;
  const button = doc.createElement('button');
  button.className = 'linkish';
  button.type = 'button';
  button.textContent = 'Try again';
  button.onclick = () => {
    button.disabled = true;
    retry();
  };
  footer.append(text, button);
}
