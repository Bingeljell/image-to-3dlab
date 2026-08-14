import * as THREE from 'three';

// The cached TRELLIS decodes are very large, indexed OBJ files containing only `v` and
// `f` records. Three.js's general OBJLoader expands every face into three new vertices;
// a 5M-face decode then needs several gigabytes of temporary JavaScript arrays. This
// loader parses the response as a stream and preserves the source indices instead.
export class IndexedOBJLoader {
  async load(url, onLoad, onProgress, onError) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      if (!response.body) throw new Error('streaming response body unavailable');

      let positions = new Float32Array(3 * 1024 * 1024);
      let indices = new Uint32Array(3 * 1024 * 1024);
      let positionCount = 0;
      let indexCount = 0;
      let vertexCount = 0;
      let loaded = 0;

      const growPositions = (needed) => {
        if (needed <= positions.length) return;
        let size = positions.length;
        while (size < needed) size *= 2;
        const next = new Float32Array(size);
        next.set(positions);
        positions = next;
      };
      const growIndices = (needed) => {
        if (needed <= indices.length) return;
        let size = indices.length;
        while (size < needed) size *= 2;
        const next = new Uint32Array(size);
        next.set(indices);
        indices = next;
      };
      const vertexIndex = (token) => {
        const raw = Number.parseInt(token.split('/', 1)[0], 10);
        return raw < 0 ? vertexCount + raw : raw - 1;
      };
      const parseLine = (line) => {
        if (line.startsWith('v ')) {
          const fields = line.trim().split(/\s+/);
          growPositions(positionCount + 3);
          positions[positionCount++] = Number.parseFloat(fields[1]);
          positions[positionCount++] = Number.parseFloat(fields[2]);
          positions[positionCount++] = Number.parseFloat(fields[3]);
          vertexCount++;
          return;
        }
        if (!line.startsWith('f ')) return;
        const fields = line.trim().split(/\s+/).slice(1);
        if (fields.length < 3) return;
        const first = vertexIndex(fields[0]);
        growIndices(indexCount + (fields.length - 2) * 3);
        for (let i = 1; i + 1 < fields.length; i++) {
          indices[indexCount++] = first;
          indices[indexCount++] = vertexIndex(fields[i]);
          indices[indexCount++] = vertexIndex(fields[i + 1]);
        }
      };

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const total = Number(response.headers.get('content-length')) || 0;
      let carry = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        loaded += value.byteLength;
        const text = carry + decoder.decode(value, { stream: true });
        const lines = text.split('\n');
        carry = lines.pop();
        for (const line of lines) parseLine(line);
        onProgress?.({ loaded, total, lengthComputable: total > 0 });
      }
      carry += decoder.decode();
      if (carry) parseLine(carry);

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position', new THREE.BufferAttribute(positions.subarray(0, positionCount), 3),
      );
      geometry.setIndex(new THREE.BufferAttribute(indices.subarray(0, indexCount), 1));
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();

      const material = new THREE.MeshStandardMaterial({ color: 0x9aa4b2, roughness: 0.85 });
      onLoad(new THREE.Mesh(geometry, material));
    } catch (error) {
      if (onError) onError(error);
      else throw error;
    }
  }
}
