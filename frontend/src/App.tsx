import { useState } from "react";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { ResearchPage } from "./pages/ResearchPage";

type Tab = "chat" | "documents" | "research";

function App() {
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-slate-900">Industrial Knowledge Intelligence OS</h1>
        <nav className="mt-3 flex gap-2">
          {(["chat", "documents", "research"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium capitalize ${
                tab === t ? "bg-indigo-600 text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>
      <main>{tab === "chat" ? <ChatPage /> : tab === "documents" ? <DocumentsPage /> : <ResearchPage />}</main>
    </div>
  );
}

export default App;
