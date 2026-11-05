/** Load a Library picture into the currently open Create sheet. */
export function createLibraryPictureLoader({ fetchPicture, freshUrl, getTarget, makeFile, onUsed = null }) {
  let request = 0;

  const invalidate = () => { request += 1; };
  const load = async (asset) => {
    const target = getTarget();
    const currentRequest = ++request;
    const source = target?.sourceRevision();
    try {
      const response = await fetchPicture(freshUrl(asset.picture));
      if (!response.ok) throw new Error(`The picture could not be read (${response.status}).`);
      const blob = await response.blob();
      if (currentRequest !== request || target !== getTarget() || !target?.canUseLibraryPicture()
        || !target.isSourceCurrent(source)) return false;
      const name = decodeURIComponent(asset.picture.split('/').pop() || 'library-picture.png');
      const used = await target.usePicture(makeFile([blob], name, { type: blob.type || 'image/png' }));
      if (used) onUsed?.(asset);
      return used;
    } catch (error) {
      if (currentRequest === request && target === getTarget() && target?.canUseLibraryPicture()
        && target.isSourceCurrent(source)) {
        target.showError(error.message || 'The picture could not be read.');
      }
      return false;
    }
  };
  return { load, invalidate };
}
