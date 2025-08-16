// FitToWidth.jsx
import React, { useRef, useState, useEffect } from "react";

/**
 * FitToWidth
 * - children: the text to fit (string or node)
 * - minFont (px) / maxFont (px): search range
 * - precision (px): stop when font-size range is within this
 * - style: extra inline styles (IMPORTANT: include fontFamily, letterSpacing, textTransform if you use them)
 * - className: wrapper classes
 */
export default function FitToWidth({
  children,
  minFont = 12,
  maxFont = 600,
  precision = 0.5,
  style = {},
  className = "",
}) {
  const containerRef = useRef(null);
  const measRef = useRef(null);
  const [fontSize, setFontSize] = useState(maxFont);

  // Fit function: binary search largest font that fits in container width
  const fit = () => {
    const container = containerRef.current;
    const meas = measRef.current;
    if (!container || !meas) return;

    const containerWidth = container.clientWidth;
    if (containerWidth <= 0) return;

    // ensure measurement element has same text & styles
    meas.textContent = typeof children === "string" ? children : "";

    let low = minFont;
    let high = maxFont;
    let mid;
    for (let i = 0; i < 30; i += 1) {
      mid = (low + high) / 2;
      meas.style.fontSize = `${mid}px`;
      // measure width
      const w = meas.scrollWidth;
      if (w > containerWidth) {
        high = mid;
      } else {
        low = mid;
      }
      if (high - low < precision) break;
    }

    // use the lower bound so it never overflows
    setFontSize(Math.max(minFont, Math.floor(low)));
  };

  // fit on mount and whenever children or style.fontFamily changes
  useEffect(() => {
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [children, style.fontFamily, style.letterSpacing, style.textTransform]);

  // fit on resize using ResizeObserver for container
  useEffect(() => {
    const ro = new ResizeObserver(() => fit());
    if (containerRef.current) ro.observe(containerRef.current);
    // also listen to window resize (fallback)
    const onResize = () => fit();
    window.addEventListener("resize", onResize);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={containerRef}
      className={`w-full ${className}`}
      style={{ position: "relative", minHeight: "1em" }}
      aria-hidden={false}
    >
      {/* visible text */}
      <div
        style={{
          fontSize: `${fontSize}px`,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          ...style,
        }}
      >
        {children}
      </div>

      {/* hidden measurement element: must match font-family / letter-spacing / text-transform */}
      <div
        ref={measRef}
        aria-hidden="true"
        style={{
          position: "absolute",
          left: -99999,
          top: 0,
          whiteSpace: "nowrap",
          visibility: "hidden",
          height: 0,
          overflow: "visible",
          ...style,
        }}
      />
    </div>
  );
}
