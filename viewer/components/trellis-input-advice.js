export function presentTrellisAdvice(result) {
  const risk = Number(result?.flat_risk);
  const pct = Number.isFinite(risk) ? `${Math.round(risk * 100)}%` : 'unknown';
  if (result?.verdict === 'likely_flat') {
    return {
      tone: 'risk', blocking: false,
      message: `TinyCLIP warning · ${pct} flat-style score. This input may produce very dark or incorrect TRELLIS materials. You can still continue.`,
    };
  }
  if (result?.verdict === 'likely_dimensional') {
    return {
      tone: 'suitable', blocking: false,
      message: `TinyCLIP check · ${pct} flat-style score. This image appears to contain useful 3D lighting cues. Please still eyeball it.`,
    };
  }
  return {
    tone: 'uncertain', blocking: false,
    message: `TinyCLIP is uncertain · ${pct} flat-style score. Please eyeball the lighting and surface shading before generating.`,
  };
}
