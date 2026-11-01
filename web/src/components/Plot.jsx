import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export const COLORS = { blue: "#1f5fa8", orange: "#e07b24", grey: "#9aa3af", green: "#2f8f5b" };

const BASE = {
  margin: { t: 36, r: 12, b: 64, l: 50 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, sans-serif", size: 12, color: "#14181f" },
  hovermode: "x unified",
  legend: { orientation: "h", y: -0.16, x: 0 },
  xaxis: { gridcolor: "#eef0f3" },
  yaxis: { gridcolor: "#eef0f3", zeroline: false },
};

/** Thin React wrapper around Plotly; `layout` is merged over the shared base style. */
export default function Plot({ data, title = "", layout = {}, height = 340 }) {
  const ref = useRef(null);

  useEffect(() => {
    // Plotly writes inferred axis types back into the layout, so each render gets a fresh copy.
    const full = { ...structuredClone(BASE), title: { text: title, font: { size: 14 }, x: 0 }, ...layout };
    Plotly.react(ref.current, data, full, { displayModeBar: false, responsive: true });
  }, [data, title, layout]);

  useEffect(() => {
    const el = ref.current;
    return () => Plotly.purge(el);
  }, []);

  return <div ref={ref} style={{ width: "100%", height }} />;
}
