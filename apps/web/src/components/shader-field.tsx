"use client";

import { MeshGradient } from "@paper-design/shaders-react";

export function ShaderField() {
  return (
    <div className="shader-field" aria-hidden="true">
      <MeshGradient
        width="100%"
        height="100%"
        colors={["#f3ead8", "#f8f4ea", "#244b9b", "#bb6d44"]}
        distortion={0.72}
        swirl={0.18}
        grainMixer={0.16}
        grainOverlay={0.1}
        speed={0.16}
        scale={1.15}
        maxPixelCount={1_200_000}
      />
      <div className="shader-wash" />
    </div>
  );
}
