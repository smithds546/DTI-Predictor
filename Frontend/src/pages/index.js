import * as React from "react"
import Layout from "../components/layout"
import Seo from "../components/seo"
import * as styles from "../components/index.module.css"

const API_BASE = process.env.GATSBY_API_BASE_URL || "http://127.0.0.1:8001"

const IndexPage = () => {
  const [drug, setDrug] = React.useState("")
  const [protein, setProtein] = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState("")
  const [result, setResult] = React.useState(null)
  const [history, setHistory] = React.useState([])

  const [drugSuggestions, setDrugSuggestions] = React.useState([]);
  const [proteinSuggestions, setProteinSuggestions] = React.useState([]);

  // Helper to highlight matched query safely
  const highlightMatch = (text = "", query = "") => {
    const idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx === -1) return text;
    return (
      text.substring(0, idx) +
      "<mark>" +
      text.substring(idx, idx + query.length) +
      "</mark>" +
      text.substring(idx + query.length)
    );
  };

  // Debounce hook
  const useDebounce = (callback, delay) => {
    const timeoutRef = React.useRef(null);
    return (...args) => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => callback(...args), delay);
    };
  };

  const updateDrugSuggestions = useDebounce(async (value) => {
    if (!value) {
      setDrugSuggestions([]);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/autocomplete/drug?q=${encodeURIComponent(value)}`);
      const backendDataRaw = await res.json();
      const backendData = Array.isArray(backendDataRaw) ? backendDataRaw : [];
      setDrugSuggestions(backendData.length > 0 ? backendData : ["No results found"]);
    } catch (err) {
      console.error("Drug autocomplete failed:", err);
      setDrugSuggestions(["No results found"]);
    }
  }, 300);

  const updateProteinSuggestions = useDebounce(async (value) => {
    if (!value) {
      setProteinSuggestions([]);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/autocomplete/protein?q=${encodeURIComponent(value)}`);
      const backendDataRaw = await res.json();
      const backendData = Array.isArray(backendDataRaw) ? backendDataRaw : [];
      setProteinSuggestions(backendData.length > 0 ? backendData : ["No results found"]);
    } catch (err) {
      console.error("Protein autocomplete failed:", err);
      setProteinSuggestions(["No results found"]);
    }
  }, 300);

  const fetchHistory = React.useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/history`)
      if (!res.ok) throw new Error("Failed to load history")
      const data = await res.json()
      setHistory(data)
    } catch (e) {
      console.error("Failed to fetch history:", e);
    }
  }, [])

  React.useEffect(() => {
    fetchHistory()
  }, [fetchHistory])

  const onSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setResult(null)
    if (!drug || !protein) {
      setError("Please enter both a drug and a protein.")
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drug, protein }),
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || "Prediction failed")
      }
      const data = await res.json()
      setResult(data)
      fetchHistory()
    } catch (e) {
      setError(e.message || "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  const onClearHistory = async () => {
    try {
      await fetch(`${API_BASE}/history`, { method: "DELETE" })
      setHistory([])
    } catch (e) {
      console.error("Failed to clear history:", e)
    }
  }

  return (
    <Layout>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <h1>Drug–Target Interaction (DTI) Checker</h1>
        <p>Enter a drug and a protein identifier/sequence to predict binder vs non-binder.</p>

        <form onSubmit={onSubmit} style={{ marginTop: 16, marginBottom: 24 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <label>
              <div>Drug</div>
              <input
                type="text"
                value={drug}
                onChange={(e) => {
                  setDrug(e.target.value);
                  updateDrugSuggestions(e.target.value);
                }}
                placeholder="e.g., Imatinib"
                style={{ width: "100%", padding: 8 }}
              />
              {drugSuggestions.length > 0 && drug && (
                <ul
                  style={{
                    border: "1px solid #888",
                    padding: 0,
                    background: "#fff",
                    marginTop: 4,
                    listStyle: "none",
                    borderRadius: 6,
                    boxShadow: "0px 2px 6px rgba(0,0,0,0.15)",
                    maxHeight: 180,
                    overflowY: "auto"
                  }}
                >
                  {drugSuggestions.map((d, idx) => (
                    <li
                      key={idx}
                      style={{ cursor: "pointer", padding: "6px 10px", borderBottom: "1px solid #eee" }}
                      onClick={() => { if(d !== "No results found") { setDrug(d); setDrugSuggestions([]); } }}
                      onMouseEnter={(e) => e.currentTarget.style.background = "#f0f4ff"}
                      onMouseLeave={(e) => e.currentTarget.style.background = "white"}
                    >
                      <span dangerouslySetInnerHTML={{ __html: d === "No results found" ? d : highlightMatch(d, drug) }} />
                    </li>
                  ))}
                </ul>
              )}
            </label>
            <label>
              <div>Protein</div>
              <input
                type="text"
                value={protein}
                onChange={(e) => {
                  setProtein(e.target.value);
                  updateProteinSuggestions(e.target.value);
                }}
                placeholder="e.g., P00533 or sequence"
                style={{ width: "100%", padding: 8 }}
              />
              {proteinSuggestions.length > 0 && protein && (
                <ul
                  style={{
                    border: "1px solid #888",
                    padding: 0,
                    background: "#fff",
                    marginTop: 4,
                    listStyle: "none",
                    borderRadius: 6,
                    boxShadow: "0px 2px 6px rgba(0,0,0,0.15)",
                    maxHeight: 180,
                    overflowY: "auto"
                  }}
                >
                  {proteinSuggestions.map((p, idx) => (
                    <li
                      key={idx}
                      style={{ cursor: "pointer", padding: "6px 10px", borderBottom: "1px solid #eee" }}
                      onClick={() => { if(p !== "No results found") { setProtein(p); setProteinSuggestions([]); } }}
                      onMouseEnter={(e) => e.currentTarget.style.background = "#f0f4ff"}
                      onMouseLeave={(e) => e.currentTarget.style.background = "white"}
                    >
                      <span dangerouslySetInnerHTML={{ __html: p === "No results found" ? p : highlightMatch(p, protein) }} />
                    </li>
                  ))}
                </ul>
              )}
            </label>
            <div>
              <button type="submit" disabled={loading} style={{ padding: "8px 16px" }}>
                {loading ? "Predicting..." : "Predict"}
              </button>
            </div>
          </div>
        </form>

        {error && (
          <div style={{ color: "#b00020", marginBottom: 16 }}>Error: {error}</div>
        )}

        {result && (
          <div style={{ border: "1px solid #ddd", padding: 16, borderRadius: 8, marginBottom: 24 }}>
            <h3>Result</h3>
            <div><b>Drug:</b> {result.drug}</div>
            <div><b>Protein:</b> {result.protein}</div>
            <div><b>Score:</b> {typeof result.score === "number" ? result.score.toFixed(4) : result.score}</div>
            <div>
              <b>Prediction:</b> {result.binder ? "Binder" : "Non-binder"}
            </div>
            <div><b>Time:</b> {new Date(result.timestamp).toLocaleString()}</div>
          </div>
        )}

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Previous Searches</h2>
          <button onClick={onClearHistory} style={{ padding: "6px 12px"}}>Clear</button>
        </div>
        <ul className={styles.list}>
          {history.length === 0 && <li>No searches yet.</li>}
          {history.map((h) => (
            <li key={h.id || h.timestamp} className={styles.listItem} style={{ display: "flex", flexDirection: "column" }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <span><b>Drug:</b> {h.drug}</span>
                <span><b>Protein:</b> {h.protein}</span>
                <span><b>Score:</b> {typeof h.score === "number" ? h.score.toFixed(4) : h.score}</span>
                <span><b>Prediction:</b> {h.binder ? "Binder" : "Non-binder"}</span>
              </div>
              <small style={{ color: "#666" }}>{new Date(h.timestamp).toLocaleString()}</small>
            </li>
          ))}
        </ul>
      </div>
    </Layout>
  )
}

export const Head = () => <Seo title="DTI Checker" />

export default IndexPage
