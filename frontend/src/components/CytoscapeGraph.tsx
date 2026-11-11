import cytoscape from "cytoscape";
import type { Core, ElementDefinition, LayoutOptions, StylesheetJsonBlock as Stylesheet } from "cytoscape";
import { useEffect, useRef } from "react";

/**
 * Shared cytoscape lifecycle wrapper (O3.4.1). Extracted from
 * GraphIntelligencePage so the Execution DAG Viewer and Root Cause Graph
 * Visualization reuse the exact same rendering mechanics instead of each
 * re-implementing cytoscape init/teardown/zoom/selection — "no duplicated
 * graph rendering" per this milestone's engineering requirements.
 *
 * Domain-specific concerns (node coloring rules, what counts as an
 * "element", any additional stylesheet rules) stay with each caller via
 * the `elements`/`stylesheet` props — this component only owns the
 * generic mechanics: container ref, cy instance lifecycle, layout,
 * zoom/fit, and selection/highlight class toggling.
 */
export interface CytoscapeGraphProps {
  elements: ElementDefinition[];
  stylesheet: Stylesheet[];
  onNodeTap?: (nodeId: string) => void;
  selectedNodeId?: string | null;
  highlightedNodeIds?: Set<string> | null;
  height?: number;
  layoutName?: string;
  testId?: string;
}

export interface CytoscapeGraphController {
  zoomBy: (factor: number) => void;
  fitToScreen: () => void;
}

export function CytoscapeGraph({
  elements,
  stylesheet,
  onNodeTap,
  selectedNodeId = null,
  highlightedNodeIds = null,
  height = 480,
  layoutName = "cose",
  testId = "graph-canvas",
}: CytoscapeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: stylesheet,
      layout: { name: layoutName, animate: false } as LayoutOptions,
      wheelSensitivity: 0.2,
    });
    if (onNodeTap) {
      cy.on("tap", "node", (event) => onNodeTap(event.target.id()));
    }
    cyRef.current = cy;
    return () => cy.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements, stylesheet, layoutName]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("selected highlighted");
    if (selectedNodeId) cy.getElementById(selectedNodeId).addClass("selected");
    if (highlightedNodeIds) {
      highlightedNodeIds.forEach((id) => cy.getElementById(id).addClass("highlighted"));
    }
  }, [selectedNodeId, highlightedNodeIds]);

  function zoomBy(factor: number) {
    const cy = cyRef.current;
    if (!cy) return;
    cy.zoom(cy.zoom() * factor);
  }

  function fitToScreen() {
    cyRef.current?.fit(undefined, 30);
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-end gap-1 text-sm">
        <button onClick={() => zoomBy(0.8)} className="rounded border border-slate-300 px-2 py-0.5" aria-label="Zoom out">
          −
        </button>
        <button onClick={() => zoomBy(1.25)} className="rounded border border-slate-300 px-2 py-0.5" aria-label="Zoom in">
          +
        </button>
        <button onClick={fitToScreen} className="ml-1 rounded border border-slate-300 px-2 py-0.5 text-xs">
          Fit to screen
        </button>
      </div>
      <div ref={containerRef} data-testid={testId} className="rounded-lg border border-slate-200 bg-slate-50" style={{ height }} />
    </div>
  );
}
