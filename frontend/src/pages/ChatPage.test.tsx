import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatPage } from "./ChatPage";
import * as api from "../services/api";
import type { ChatResponse } from "../services/api";

vi.mock("../services/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/api")>();
  return { ...actual, sendChatQuery: vi.fn() };
});

const sendChatQuery = vi.mocked(api.sendChatQuery);

afterEach(() => {
  vi.clearAllMocks();
});

function baseResponse(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    summary: "Pump P-101 failed due to cavitation.",
    confidence: 0.8,
    citations: [],
    evidence: [],
    recommended_actions: [],
    reasoning_summary: "Analyzed 1 piece(s) of evidence.",
    contradictions: [],
    recommendations: [],
    latencies: {},
    claims: [],
    faithfulness: null,
    sentence_attributions: [],
    ...overrides,
  };
}

async function ask(query = "Why did P-101 fail?") {
  const user = userEvent.setup();
  render(<ChatPage />);
  await user.type(screen.getByPlaceholderText(/Ask about equipment/), query);
  await user.click(screen.getByRole("button", { name: "Ask" }));
}

describe("ChatPage — O2 faithfulness/claim graph/reasoning path", () => {
  it("renders the faithfulness score and summary", async () => {
    sendChatQuery.mockResolvedValue(
      baseResponse({
        faithfulness: {
          overall_faithfulness: 0.82,
          sentence_faithfulness: [],
          recommendation_faithfulness: 1.0,
          graph_faithfulness: 0.9,
          claim_coverage: 0.75,
          unsupported_claim_count: 1,
          hallucinated_claim_count: 0,
          missing_evidence_count: 0,
          contradiction_rate: 0.0,
          summary: "Faithfulness 0.82: 1/1 sentence(s) entailed by a verified claim.",
        },
      }),
    );

    await ask();

    expect(await screen.findByText("Faithfulness")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
    expect(screen.getByText(/Faithfulness 0.82: 1\/1 sentence/)).toBeInTheDocument();
  });

  it("renders claim graph entries with verification status and confidence", async () => {
    sendChatQuery.mockResolvedValue(
      baseResponse({
        claims: [
          {
            id: "c1",
            claim_type: "fact",
            text: "Equipment = Pump P-101",
            subject: "P-101",
            predicate: "is_a",
            object: "pump",
            confidence: 0.6,
            propagated_confidence: 0.7,
            source_evidence_ids: ["chunk-1"],
            verification_status: "supported",
            verification_notes: "Confirmed in the Knowledge Graph.",
            support_count: 2,
            contradiction_count: 0,
          },
        ],
      }),
    );

    await ask();

    expect(await screen.findByText("Claim Graph")).toBeInTheDocument();
    expect(screen.getByText("Equipment = Pump P-101")).toBeInTheDocument();
    expect(screen.getByText("supported")).toBeInTheDocument();
    expect(screen.getByText(/confidence 0.60/)).toBeInTheDocument();
    expect(screen.getByText(/propagated 0.70/)).toBeInTheDocument();
  });

  it("renders reasoning path attribution behind an expandable panel", async () => {
    sendChatQuery.mockResolvedValue(
      baseResponse({
        sentence_attributions: [
          {
            sentence: "Pump P-101 failed due to cavitation.",
            supporting_claim_ids: ["c1"],
            supporting_evidence_ids: ["chunk-1"],
            graph_node_ids: ["node-1"],
            graph_edge_descriptions: ["c1 --CAUSES--> c2"],
            confidence: 0.75,
            reasoning_path: "Evidence[chunk-1] -> Claim[c1](supported) -> Answer sentence",
          },
        ],
      }),
    );

    await ask();

    expect(await screen.findByText("Reasoning Path")).toBeInTheDocument();
    expect(screen.getByText("Inspect reasoning path")).toBeInTheDocument();
  });

  it("shows a verification badge on each recommendation", async () => {
    sendChatQuery.mockResolvedValue(
      baseResponse({
        recommendations: [
          {
            id: "r1",
            description: "Investigate bearing wear",
            priority: "high",
            severity: "high",
            estimated_impact: "downtime risk",
            required_personnel: ["Technician"],
            required_equipment: [],
            required_documents: [],
            estimated_downtime: "TBD",
            confidence: 0.8,
            justification: "Derived from the reasoning graph.",
            supporting_evidence: ["chunk-1"],
            verification_status: "verified",
            verification_reasons: ["Backed by 1 evidence item(s), confidence 0.8, and 1 verified claim(s)."],
          },
        ],
      }),
    );

    await ask();

    expect(await screen.findByText("Investigate bearing wear")).toBeInTheDocument();
    expect(screen.getByText("verified")).toBeInTheDocument();
  });

  it("omits the O2 sections entirely when the response carries no claims/faithfulness/attributions", async () => {
    sendChatQuery.mockResolvedValue(baseResponse());

    await ask();

    expect(await screen.findByText("Answer")).toBeInTheDocument();
    expect(screen.queryByText("Faithfulness")).not.toBeInTheDocument();
    expect(screen.queryByText("Claim Graph")).not.toBeInTheDocument();
    expect(screen.queryByText("Reasoning Path")).not.toBeInTheDocument();
  });
});
