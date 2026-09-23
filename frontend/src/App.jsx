import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const examples = [
  "exam tomorrow and I haven't studied",
  "my friend borrowed money and didn't return it",
  "everyone is studying except me",
  "I finished my project successfully",
];

function App() {
  const [query, setQuery] = useState("");
  const [memes, setMemes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const searchMemes = async () => {
    if (!query.trim()) {
      return;
    }

    try {
      setLoading(true);
      setError("");

      const response = await axios.get(
        `${API_URL}/api/search`,
        {
          params: {
            query: query.trim(),
            limit: 5,
          },
        }
      );

      setMemes(response.data.results);
    } catch (err) {
      console.error("Search error:", err);

      setMemes([]);
      setError(
        "Unable to connect to MemeAI backend. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const selectExample = (example) => {
    setQuery(example);
    setError("");
  };

  return (
    <div className="app">
      <header className="navbar">
        <div className="logo">MemeAI</div>

        <div className="nav-text">
          AI-Powered Meme Search
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <p className="eyebrow">
            SEMANTIC MEME SEARCH
          </p>

          <h1>
            Find the meme that
            <br />
            matches your thought.
          </h1>

          <p className="description">
            Describe what you are feeling or looking for,
            and MemeAI finds the most relevant memes using
            AI-powered semantic search.
          </p>

          <div className="search-box">
            <input
              type="text"
              placeholder="Example: exam tomorrow and I haven't studied"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  searchMemes();
                }
              }}
            />

            <button
              onClick={searchMemes}
              disabled={loading || !query.trim()}
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </div>

          <div className="examples">
            <span>Try:</span>

            {examples.map((example) => (
              <button
                key={example}
                onClick={() => selectExample(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {loading && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Finding matching memes...</p>
          </div>
        )}

        {!loading && memes.length > 0 && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <p className="results-label">
                  SEARCH RESULTS
                </p>

                <h2>Matching Memes</h2>
              </div>

              <span>
                {memes.length} results
              </span>
            </div>

            <div className="meme-grid">
              {memes.map((meme, index) => (
                <article
                  className="meme-card"
                  key={`${meme.image}-${index}`}
                >
                  <div className="image-container">
                    <img
                      src={`${API_URL}${meme.image_url}`}
                      alt={meme.caption || "Meme"}
                      loading="lazy"
                    />
                  </div>

                  <div className="meme-info">
                    <div className="rank-row">
                      <span className="rank">
                        #{index + 1}
                      </span>

                      <span className="match">
                        {(meme.similarity * 100).toFixed(1)}% match
                      </span>
                    </div>

                    <p className="caption">
                      {meme.caption}
                    </p>

                    <div className="meta">
                      <span>
                        {meme.sentiment}
                      </span>

                      <span>
                        {meme.image}
                      </span>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        {!loading &&
          memes.length === 0 &&
          !error && (
            <section className="empty-state">
              <div className="empty-icon">
                ✦
              </div>

              <h3>
                Search for a meme
              </h3>

              <p>
                Describe a situation, emotion, or idea
                in your own words.
              </p>
            </section>
          )}
      </main>
    </div>
  );
}

export default App;