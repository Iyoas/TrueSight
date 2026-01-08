"use client";

import React, { useRef, useState } from "react";

type CueImageMagnifierProps = {
  src: string;
  alt: string;
  zoom?: number;
  lensSize?: number;
  className?: string;
};

const CueImageMagnifier: React.FC<CueImageMagnifierProps> = ({
  src,
  alt,
  zoom = 2.5,
  lensSize = 120,
  className = "",
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const lensRef = useRef<HTMLDivElement>(null);
  const [isActive, setIsActive] = useState(false);

  const updateLens = (event: React.MouseEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    const lens = lensRef.current;
    if (!container || !lens) {
      return;
    }

    const rect = container.getBoundingClientRect();
    const radius = lensSize / 2;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const clampedX = Math.max(radius, Math.min(x, rect.width - radius));
    const clampedY = Math.max(radius, Math.min(y, rect.height - radius));
    const backgroundX = -(clampedX * zoom - radius);
    const backgroundY = -(clampedY * zoom - radius);

    lens.style.left = `${clampedX - radius}px`;
    lens.style.top = `${clampedY - radius}px`;
    lens.style.width = `${lensSize}px`;
    lens.style.height = `${lensSize}px`;
    lens.style.backgroundImage = `url("${src}")`;
    lens.style.backgroundSize = `${rect.width * zoom}px ${rect.height * zoom}px`;
    lens.style.backgroundPosition = `${backgroundX}px ${backgroundY}px`;
  };

  return (
    <div
      ref={containerRef}
      className={`cue-magnifier${isActive ? " cue-magnifier--active" : ""} ${className}`.trim()}
      onMouseEnter={(event) => {
        setIsActive(true);
        updateLens(event);
      }}
      onMouseMove={updateLens}
      onMouseLeave={() => setIsActive(false)}
    >
      <img src={src} alt={alt} className="cue-image cue-magnifier__image" />
      <div ref={lensRef} className="cue-magnifier__lens" />
    </div>
  );
};

export default CueImageMagnifier;
