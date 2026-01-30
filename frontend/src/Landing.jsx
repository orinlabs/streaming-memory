import { useEffect, useState } from "react";

import { AnimatePresence, motion } from "framer-motion";
import { Link } from "react-router-dom";

// Animated streaming memory visualization
function StreamingMemoryVisual() {
  const [step, setStep] = useState(0);

  const memories = [
    "Dad's birthday is March 15th",
    "Loves playing golf on Saturdays",
    "Frustrated with his driver",
    "Mom got him a watch already",
    "Golf buddies are Tom and Jerry",
    "Wants to play Pebble Beach",
    "His handicap is around 18",
    "Saw Callaway ad during PGA",
  ];

  const activeAtStep = [
    [0, 1, 2],
    [1, 2, 5],
    [2, 5, 7],
    [2, 6, 7],
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((s) => (s + 1) % activeAtStep.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  const activeMemories = activeAtStep[step];

  return (
    <div className="flex items-center gap-6 py-8">
      {/* Memory Pool */}
      <div className="flex-1">
        <div className="text-xs text-[#999] mb-3 uppercase tracking-wide">
          Memory Pool
        </div>
        <div className="grid grid-cols-2 gap-2">
          {memories.map((memory, i) => (
            <motion.div
              key={i}
              animate={{
                opacity: activeMemories.includes(i) ? 1 : 0.3,
                scale: activeMemories.includes(i) ? 1 : 0.95,
              }}
              transition={{ duration: 0.3 }}
              className={`text-xs p-2 rounded border ${
                activeMemories.includes(i)
                  ? "border-[#1a1a1a] bg-[#fafafa] text-[#1a1a1a]"
                  : "border-[#eee] text-[#999]"
              }`}
            >
              {memory}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Arrow */}
      <div className="flex flex-col items-center gap-1 text-[#ccc]">
        <svg
          className="w-8 h-8"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M14 5l7 7m0 0l-7 7m7-7H3"
          />
        </svg>
        <span className="text-xs">streaming</span>
      </div>

      {/* Context Window */}
      <div className="w-48">
        <div className="text-xs text-[#999] mb-3 uppercase tracking-wide">
          Context Window
        </div>
        <div className="border border-[#1a1a1a] rounded-lg p-3 bg-[#fafafa]">
          <AnimatePresence mode="popLayout">
            {activeMemories
              .sort((a, b) => a - b)
              .map((nodeIndex) => (
                <motion.div
                  key={nodeIndex}
                  layout
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                  className="text-xs text-[#1a1a1a] py-1"
                >
                  • {memories[nodeIndex]}
                </motion.div>
              ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

// Token efficiency comparison graph
function EfficiencyGraph() {
  const logY = (tokens) => {
    const logVal = Math.log10(tokens);
    return 170 - ((logVal - 3) / 2) * 140;
  };

  return (
    <div className="py-8">
      <div className="text-xs text-[#999] w-full text-center mb-4 uppercase tracking-wide">
        Context Size During Generation
      </div>
      <svg viewBox="0 0 400 200" className="w-full h-48">
        <line x1="50" y1="30" x2="50" y2="170" stroke="#eee" strokeWidth="1" />
        <line x1="50" y1="170" x2="380" y2="170" stroke="#eee" strokeWidth="1" />

        {[1000, 10000, 100000].map((val) => (
          <line
            key={val}
            x1="50"
            y1={logY(val)}
            x2="380"
            y2={logY(val)}
            stroke="#f5f5f5"
            strokeWidth="1"
          />
        ))}

        <text
          x="45"
          y={logY(100000) + 4}
          textAnchor="end"
          className="text-[10px] fill-[#999]"
        >
          100k
        </text>
        <text
          x="45"
          y={logY(10000) + 4}
          textAnchor="end"
          className="text-[10px] fill-[#999]"
        >
          10k
        </text>
        <text
          x="45"
          y={logY(1000) + 4}
          textAnchor="end"
          className="text-[10px] fill-[#999]"
        >
          1k
        </text>

        <text
          x="215"
          y="190"
          textAnchor="middle"
          className="text-[10px] fill-[#999]"
        >
          Generated tokens →
        </text>

        <path
          d={`M 50 ${logY(80000)} L 380 ${logY(80000)}`}
          fill="none"
          stroke="#f87171"
          strokeWidth="2"
        />
        <path
          d={`M 50 ${logY(3000)} Q 150 ${logY(5000)} 220 ${logY(8000)} T 380 ${logY(15000)}`}
          fill="none"
          stroke="#fb923c"
          strokeWidth="2"
        />
        <path
          d={`M 50 ${logY(2000)} L 380 ${logY(2000)}`}
          fill="none"
          stroke="#22c55e"
          strokeWidth="2"
        />
      </svg>

      <div className="flex justify-center gap-6 mt-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-red-400" />
          <span className="text-[#666]">Prompt stuffing</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-orange-400" />
          <span className="text-[#666]">Agent RAG</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-green-500" />
          <span className="text-[#666]">Streaming memory</span>
        </div>
      </div>
    </div>
  );
}

export default function Landing() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-2xl mx-auto px-6">
        {/* Hero */}
        <section className="pt-24 pb-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl md:text-5xl font-bold text-[#1a1a1a] mb-6 leading-tight">
              Memory at the
              <br />
              Speed of Thought
            </h1>

            <p className="text-lg text-[#666] mb-4">
              We introduce "streaming memory"—an approach that lets models do
              multi-hop reasoning over large memory banks in a single turn
              without exploding context.
            </p>

            <Link
              to="/demo"
              className="inline-block px-8 py-4 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors text-lg"
            >
              Try the Demo
            </Link>
          </motion.div>
        </section>

        {/* The Problem */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              The Problem
            </h2>

            <p className="text-lg text-[#666] mb-4">
              When building a memory system, you face a critical question: do
              you want to decide what's important to remember at ingestion time
              or query time?
            </p>

            <p className="text-lg text-[#666] mb-4">
              Deciding at ingestion time leads to inaccuracies. What if you
              didn't know that a certain fact was important when you saw it, but
              later context revealed that it was? Too bad.
            </p>

            <p className="text-lg text-[#666] mb-4">
              Deciding at query time rewards remembering as much as possible,
              then searching the memory when needed. But writing a good query
              function for an arbitrarily large memory bank is very difficult.
              It's also slow—the agent will need to make multiple queries over
              multiple LLM turns. And it's token inefficient, since the agent
              will inevitably query memories it doesn't need.
            </p>

            <p className="text-lg text-[#666]">
              Streaming memory combines the best of both worlds. It allows
              abnormally large memory banks while also being fast to reason over
              and extremely token efficient.
            </p>
          </motion.div>
        </section>

        {/* How It Works */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              How It Works
            </h2>

            <p className="text-lg text-[#666] mb-4">
              Using the last X tokens of a partial reasoning trace as a vector,
              we query a memory bank using an arbitrary query function. We then
              inject the top reranked memories at the start of the context,
              replacing the old ones.
            </p>

            <p className="text-lg text-[#666] mb-4">
              We rerank the memories every Y output tokens, causing the entire
              prompt to update many times within a single reasoning trace.
            </p>

            <StreamingMemoryVisual />

            <p className="text-lg text-[#666] mb-4">
              This allows models to do multi-hop reasoning over the memory bank.
              As the model explores sample space, the relevant memories will
              change and refine, allowing it to explore further without
              polluting the context.
            </p>

            <p className="text-lg text-[#666]">
              Instead of the context window being the maximum total amount of
              memories a model can reason over, streaming memory allows the
              model to reason over an infinite memory pool—just not all at once.
            </p>

            <EfficiencyGraph />
          </motion.div>
        </section>

        {/* Limitations */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              Limitations
            </h2>

            <p className="text-lg text-[#666] mb-4">
              The hard part with this approach, like others, is writing a good
              query function.
            </p>

            <p className="text-lg text-[#666] mb-4">
              But the streaming mechanism is independent of both the query
              function and the memory bank structure, meaning you can swap in
              different retrieval methods without changing how streaming works.
            </p>

            <p className="text-lg text-[#666]">
              However, models will sometimes commit too hard to a given line of
              reasoning and not be able to adapt based on new memories that
              surface. The model, when in late-stage reasoning, is not aware
              that it had different memories earlier in the reasoning.
              Backtracking is not yet a learned behavior.
            </p>
          </motion.div>
        </section>

        {/* Demo */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">Demo</h2>

            <p className="text-lg text-[#666] mb-8">
              Our demo uses Qwen3-8B and usually has {"<"}700 tokens in context
              despite a memory bank of ~15k tokens.
            </p>

            <Link
              to="/demo"
              className="inline-block px-8 py-4 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors"
            >
              Try it here
            </Link>

            <div className="flex items-center gap-6 text-xs text-[#999] mt-8">
              <a
                href="https://github.com/acadia-learning/streaming-memory"
                target="_blank"
                rel="noopener"
                className="hover:text-[#666] transition-colors"
              >
                GitHub
              </a>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
  );
}
